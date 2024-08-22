import torch
import torch.nn as nn
from net import Mei, DecoderBlock
from diffusers import AutoencoderKL
from clipEmbedding import CLIPEncoder
import torchvision
from torchvision.utils import save_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder = "vae").to(device)
vae.eval()

SCALING_FACTOR = vae.config.scaling_factor

def decode_img(latents):
    latents = latents / SCALING_FACTOR
    with torch.no_grad():
        samples = vae.decode(latents).sample
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    image = Image.fromarray(samples)
    # image = image.detach()
    return image
# decoder = DecoderBlock(input_size = 32, patch_size = 2, in_channels = 4, hidden_size = 768, depth = 24, num_heads = 512, mlp_ratio = 4.0, drop_path = 0, window_size = 0, window_block_indexes = None, use_rel_pos = False, caption_channels = 512, lewei_scale = 1.0, config = None, model_max_length = 77)
model = Mei(input_size = 32, patch_size = 2, in_channels = 4, hidden_size = 768, depth = 24, output_dim=512, num_heads = 16).to(device)
model.eval()
checkpoint = torch.load("/home/a2soni/Mei/model_checkpoints_autoenc_img_loss/checkpoints_steps/epoch_5_step_150000.pth", map_location="cpu")

state_dict = checkpoint.get('state_dict', checkpoint)
missing, unexpect = model.load_state_dict(state_dict, strict=False)

model = model.to(device)

latent_x1 = torch.load("/home/a2soni/latents_x1/000000001_latent.pt").to(device)
# latent_x1 = torch.randn(1, 4, 32, 32).to(device)
# print(latent_x1.view(latent_x1.size(0), -1).min(dim = 1, keepdim = True).values)

# print(latent_x1.view(latent_x1.size(0), -1).min(dim = 1, keepdim = True).values.shape)
# latents: Original latent values
# Scaling the latents to the range [-1, 1]

# min_val = latent_x1.view(latent_x1.size(0), -1).min(dim=1).values
# max_val = latent_x1.view(latent_x1.size(0), -1).max(dim=1).values

# latents_scaled = ((latent_x1 - min_val[:, None, None, None]) / (max_val[:, None, None, None] - min_val[:, None, None, None])) * 2 - 1

# Assuming latent_x1 is your tensor of latents
# mean_val = latent_x1.view(latent_x1.size(0), -1).mean(dim=1)
# std_val = latent_x1.view(latent_x1.size(0), -1).std(dim=1)

# Normalize latents
# latents_normalized = (latent_x1 - mean_val[:, None, None, None]) / std_val[:, None, None, None]

# Use latents_normalized for further processing or training

# latents = ((latent_x1 - latent_x1.view(latent_x1.size(0), -1).min(dim = 1).values)/(latent_x1.view(latent_x1.size(0), -1).max(dim = 1).values - latent_x1.view(latent_x1.size(0), -1).min(dim = 1).values))*2 - 1
# print(latents_normalized.shape)
# print(latent_x1)
# print(latents.max(),latents.min())

# # caption = "A lady in a checkered blue shirt strikes a pose confidently."
original_caption = "a white cupboard"
clipenc = CLIPEncoder("cuda")
# embed = clipenc.get_text_embeddings([original_caption])
clipemb, attn_mask = clipenc.get_text_embeddings([original_caption])
y = clipemb.text_embeds
y_t = clipemb.last_hidden_state
y = y.unsqueeze(1).unsqueeze(2).to(device)
y_t = y_t.unsqueeze(1).to(device)
# # embed_prev, _, avg_embedding_prev = t5Embed.get_text_embeddings([original_caption])

# embed = embed.unsqueeze(1).unsqueeze(2).to(device)
# # att_mask = att_mask.unsqueeze(1).unsqueeze(2).to(device)
# # avg_embedding = avg_embedding.unsqueeze(1).unsqueeze(2).to(device)

# # embed_prev = embed_prev.unsqueeze(1).to(device)
# # avg_embedding_prev = avg_embedding_prev.unsqueeze(1).unsqueeze(2).to(device)


with torch.no_grad():
    predicted_latent, mu, logvar, z = model(latent_x1, y, y_t, attn_mask)

# predicted_latent: Predicted scaled latent values in the range [-1, 1]
# Reverting the scaling back to the original range

# predicted_latent_unscaled = ((predicted_latent + 1) / 2) * (max_val[:, None, None, None] - min_val[:, None, None, None]) + min_val[:, None, None, None]

print(predicted_latent.max(),predicted_latent.min())
# predicted_latent = ((predicted_latent +1)/2)*(latent_x1.max(dim = 1).values - latent_x1.min(dim = 1).values) + latent_x1.min(dim = 1).values
img = decode_img(predicted_latent)
original_img = decode_img(latent_x1)
# img = torchvision.transforms.functional.to_pil_image(img)
img = img.save("latest_model_generated_06.jpg")
original_img = original_img.save("original_image_06.jpg")

print(mu.mean())
print(logvar.mean())
