import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, Attention as Attention_, PatchEmbed
from timm.models.layers import DropPath
import xformers.ops
import numpy as np
from collections.abc import Iterable
from itertools import repeat
from clipEmbedding import CLIPEncoder

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

class WindowAttention(Attention_):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        if use_fp32_attention := getattr(self, 'fp32_attention', False):
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CaptionEmbedder(nn.Module):
    """
    Embeds text caption(condition) into vector representations.
    """

    def __init__(self, in_channels, hidden_size, act_layer=nn.GELU(approximate='tanh')):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(1, in_channels) / in_channels ** 0.5))

    def forward(self, caption, train):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        caption = self.y_proj(caption)
        return caption

class EncoderFinalLayer(nn.Module):
    """
    The final layer of Mei Encoder.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, y):
        shift, scale = (self.scale_shift_table[None] + y[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

#upsampling latent code z, so that it can be passed through decoder transformer blocks

class DecoderUpsampleBlock(nn.Module):
    def __init__(self, hidden_size, patch_size, output_size, out_channels):
        super().__init__()
        self.T = int((output_size*output_size) / patch_size**2)
        self.interpolation_size = int(patch_size*patch_size*out_channels*self.T)
        self.embed_latent = nn.Linear(patch_size*patch_size*out_channels, hidden_size, bias=True)
    
    def forward(self, z):
        B = z.shape[0]
        z = z.unsqueeze(1)
        z = F.interpolate(z, size=(self.interpolation_size, ), mode="linear", align_corners=False)
        z = z.squeeze(1)
        z = z.view(B, self.T, -1)
        z = self.embed_latent(z)
        return z

class DecoderFinalLayer(nn.Module):
    """
    The final layer of Mei Decoder.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, y):
        shift, scale = (self.scale_shift_table[None] + y[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x      

class TransformerBlock(nn.Module):
    """
    A Transformer block with adaptive layer norm (adaLN-single) conditioning.
    It will be a base of Encoder and Decoder Module
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + y.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x

class EncoderBlock(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, output_dim = 512, depth=28, num_heads=16, mlp_ratio=4.0, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=512, lewei_scale=1.0, config=None, model_max_length=77, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.in_channels = in_channels
        #keeping consistent with pixart-alpha code, but pred_sigma will be always false as we are not predicting noise, but latent space
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale
        self.output_dim = output_dim
        self.T  = int((input_size*input_size) / self.patch_size**2) 

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, act_layer=approx_gelu)

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size // patch_size, input_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False)
            for i in range(depth)
        ])
        self.final_layer = EncoderFinalLayer(hidden_size, patch_size, self.out_channels)
        
        
        self.mu_linear = nn.Linear(patch_size * patch_size * self.out_channels * self.T, output_dim, bias=True)
        self.logvar_linear = nn.Linear(patch_size * patch_size * self.out_channels * self.T, output_dim, bias=True)
        self.model_max_len = model_max_length
        self.initialize_weights()

    def forward(self, x, y, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        y: (N, 1, 1, C) tensor of caption
        """
        B, C, H, W = x.shape
        x = x.to(self.dtype)
        y = y.to(self.dtype)

        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        y = self.y_embedder(y, self.training)  # (N, 1, 1, D)
 
        y = y.squeeze(1).squeeze(1)

        y_addn = self.y_block(y) 
        for block in self.blocks:
            x = block(x, y_addn)  # (N, T, D) 
        # print(f"transformer output shape: {x.shape}")
        x = self.final_layer(x, y)  # (N, T, patch_size ** 2 * out_channels)
        # print(f"final layer shape: {x.shape}")
        
        #flatting X for mu and logvar calculation
        x = x.view(B, -1)
        mu = self.mu_linear(x)
        log_var = self.logvar_linear(x)
        return mu, log_var

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.y_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        nn.init.xavier_uniform_(self.mu_linear.weight)
        nn.init.constant_(self.mu_linear.bias, 0)
        
        nn.init.xavier_uniform_(self.logvar_linear.weight)
        nn.init.constant_(self.logvar_linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
class DecoderBlock(nn.Module):
    """
    Decoder block with a Transformer backbone.
    """

    def __init__(self, output_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=512, lewei_scale=1.0, config=None, model_max_length=77, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.out_size = output_size
        self.in_channels = in_channels
        #keeping consistent with pixart-alpha code, but pred_sigma will be always false as we are not predicting noise, but latent space
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale 
        self.base_size = output_size // self.patch_size
        
        self.num_patches = int((output_size*output_size) / patch_size**2)

        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, self.num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.decoder_upsample = DecoderUpsampleBlock(hidden_size, patch_size, output_size, self.out_channels)
        self.y_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, act_layer=approx_gelu)

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(output_size // patch_size, output_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False)
            for i in range(depth)
        ])
        self.final_layer = DecoderFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def forward(self, x, y, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, 512) latent code
        y: (N, 1, 1, C) tensor of caption
        """
        x = x.to(self.dtype)
        y = y.to(self.dtype)

        pos_embed = self.pos_embed.to(self.dtype)
        x = self.decoder_upsample(x)
        # print(f"after latent upsample : {x.shape}")
        x = x + pos_embed
        
        y = self.y_embedder(y, self.training)  # (N, 1, 1, D)
 
        y = y.squeeze(1).squeeze(1)

        y_addn = self.y_block(y) 
        
        for block in self.blocks:
            x = block(x, y_addn)  # (N, T, D) 
            
        # print(f"after decoder transformer blocks: {x.shape}")
            
        x = self.final_layer(x, y)  # (N, T, patch_size ** 2 * out_channels)
        # print(f"decoder final layer shape: {x.shape}")
        
        x = self.unpatchify(x)
        
        return x

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.y_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        nn.init.xavier_uniform_(self.decoder_upsample.embed_latent.weight)
        nn.init.constant_(self.decoder_upsample.embed_latent.bias, 0)
    

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class Mei(nn.Module):
    def __init__(self, input_size = 32, patch_size = 2, in_channels=4, hidden_size=1152, output_dim = 512, depth=28, num_heads=16, mlp_ratio=4.0, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=512, lewei_scale=1.0, config=None, model_max_length=77):
        super().__init__()
        self.encoder = EncoderBlock(input_size, patch_size, in_channels, hidden_size, output_dim, depth, num_heads, mlp_ratio, drop_path, window_size, window_block_indexes, use_rel_pos, caption_channels, lewei_scale, config, model_max_length)
        self.decoder = DecoderBlock(input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio, drop_path, window_size, window_block_indexes, use_rel_pos, caption_channels, lewei_scale, config, model_max_length)
    
    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        std = torch.exp(0.5 * logvar)
        z = mu + std*torch.randn_like(std)
        x = self.decoder(z, y)
        return x, mu, std, z
    

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mei(input_size = 32, patch_size = 2, in_channels = 4, hidden_size = 768, depth = 12, output_dim=512, num_heads = 16).to(device)
    latent_image_batch = torch.rand(3, 4, 32, 32).to(device)
    texts = ["text zeta", "write beta", "play alpha"]

    clipEncoder = CLIPEncoder("cuda")
    y = clipEncoder.get_text_embeddings(texts)


    y = y.unsqueeze(1).unsqueeze(2).to(device)

    x = model(latent_image_batch, y)
    print(x.shape)