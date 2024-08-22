import os
import sys
import math
import types
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import argparse
import datetime
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore") 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from transformers import AdamW, get_constant_schedule_with_warmup
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from accelerate.logging import get_logger
from mmcv.runner import LogBuffer
# from diffusers import AutoencoderKL
from copy import deepcopy
from PIL import Image
import numpy as np
from clipEmbedding import CLIPEncoder
from net import Mei
from loss import MutualInformationLoss, ods_loss, CCALoss, ODSLoss, MMDLoss, KLDivergence
from dist_utils import synchronize, get_world_size, clip_grad_norm_
import wandb
from utils import init_random_seed, set_random_seed, save_checkpoint, load_checkpoint

torch.multiprocessing.set_start_method('spawn', force=True)

#config
LATENT_DIR_X1 = "/home/a2soni/latents_x1"
VAE_MODEL_PATH = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
# LATENT_DIR_X2 = "/root/achint/InvertEdit/latents_directory"
GRADIENT_CLIP = 10
EMA_RATE = 0.999
LOG_INTERVAL = 500
SAVE_MODEL_STEPS = 50000
SAVE_MODEL_EPOCHS = 1
OUTPUT_DIR = "/home/a2soni/Mei/model_checkpoints_autoenc_img_loss"
MIXED_PRECISION = 'fp16'
GRADIENT_ACCUMULATION_STEPS = 4
SEED = 7
DATA_PATH = "/home/a2soni/final_latents.parquet"
TRAIN_BATCH_SIZE = 128
WARMUP_STEPS = 2000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 3e-2
MODEL_NAME = "disEntangleEditBase"
NUM_EPOCHS = 100
MSE_SCALE = 2

def decode_img(latents, vae, SCALING_FACTOR):
    # SCALING_FACTOR = vae.config.scaling_factor
    latents = latents / SCALING_FACTOR
    with torch.no_grad():
        samples = vae.decode(latents).sample
    return samples

# defining accelerator logger
logger = get_logger(__name__)

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'DisentangleBlock' #need to put layer consisting of Attention blocks


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.5-0.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L 

def train(model, model_ema, num_epochs, train_dataloader, clipModel, optimizer, lr_scheduler, start_epoch = 0):
    time_start, last_tic = time.time(), time.time()
    
    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * num_epochs
    log_buffer = LogBuffer()
    # mse = MSELoss(reduction = '')
    # l1 = L1Loss(reduction = 'mean')
    # mutualInfoLoss = MutualInformationLoss()
    # mmdloss = MMDLoss(device = current_device)
    kldloss = KLDivergence()
    # beta_raw = torch.tensor(-6.0, requires_grad=True)
    # optimizer.add_param_group({'params': beta_raw})
    # ccaloss = CCALoss()
    total_steps = len(train_dataloader) * num_epochs
    cycles_per_epoch = 5
    total_cycles = num_epochs * cycles_per_epoch
    beta_np = frange_cycle_cosine(0, 2, total_steps, total_cycles, 0.25)
    beta = torch.tensor(beta_np, dtype=torch.float32).to(accelerator.device)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        print(f"epoch : {epoch} started")
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start

            # latent_x1 = ((batch["latent_x1"] - batch["latent_x1"].min(dim = 1).values)/(batch["latent_x1"].max(dim = 1).values - batch["latent_x1"].min(dim = 1).values))*2 - 1
            caption_text = batch["original_caption"]

            #getting text embeddings : prev_caption is used for passing embedding representation difference in adaptive group normalization layer
            clipemb, attn_mask = clipModel.get_text_embeddings(caption_text)
            
            y = clipemb.text_embeds
            y_t = clipemb.last_hidden_state
            
            #maintaing model required dimensions
            latent_x1 = batch["latent_x1"]
            latent_x1 = latent_x1.squeeze(1)
            
            y = y.unsqueeze(1).unsqueeze(2)
            y_t = y_t.unsqueeze(1)
            # print(attn_mask.shape)
            grad_norm = None
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predicted_latent, mu, logvar, z = model(latent_x1, y, y_t, attn_mask)
                
                # predicted_images = decode_img(predicted_latent, vae, SCALING_FACTOR)
                # original_images = decode_img(latent_x1, vae, SCALING_FACTOR)
                
                # print(original_images.shape, predicted_images.shape)
                
                ###Experiments###
                # mse_loss = F.mse_loss(predicted_images, original_images, size_average=False).div(original_images.shape[0])
                latent_loss = F.mse_loss(predicted_latent, latent_x1, size_average=False).div(latent_x1.shape[0])
                
                latent_l1_loss = F.l1_loss(latent_x1, predicted_latent, size_average = False).div(latent_x1.shape[0])
                kld_loss = kldloss(mu, logvar)
                loss = latent_l1_loss + 0.1*latent_loss + beta[global_step]*kld_loss
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients:
                    ema_update(model_ema, model, EMA_RATE)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {"train_loss": accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            # logs.update(mmd_loss = accelerator.gather(loss).mean().item())
            logs.update(beta = accelerator.gather(beta[global_step]).mean().item())
            logs.update(latent_l1_loss = accelerator.gather(latent_l1_loss).mean().item())
            logs.update(kld_loss = accelerator.gather(kld_loss).mean().item())
            logs.update(latent_mse_loss = accelerator.gather(latent_loss).mean().item())
            log_buffer.update(logs)

            # logging on terminal
            if (step + 1) % LOG_INTERVAL == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / LOG_INTERVAL
                t_d = data_time_all / LOG_INTERVAL
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                
                info = f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}), "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info, main_process_only=True)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0

            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            global_step += 1
            data_time_start= time.time()

            synchronize()
            if accelerator.is_main_process:
                if ((epoch - 1) * len(train_dataloader) + step + 1) % SAVE_MODEL_STEPS == 0:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(OUTPUT_DIR, 'checkpoints_steps'),
                                    epoch=epoch,
                                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                                    model=accelerator.unwrap_model(model),
                                    model_ema=accelerator.unwrap_model(model_ema),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
            synchronize()

        synchronize()
        if accelerator.is_main_process:
            if epoch % SAVE_MODEL_EPOCHS == 0 or epoch == num_epochs:
                os.umask(0o000)
                save_checkpoint(os.path.join(OUTPUT_DIR, 'checkpoints'),
                                epoch=epoch,
                                step=(epoch - 1) * len(train_dataloader) + step + 1,
                                model=accelerator.unwrap_model(model),
                                model_ema=accelerator.unwrap_model(model_ema),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
                        
        synchronize()

class DisEntangleDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_parquet(data_path, engine="pyarrow")
    
    def __len__(self):
        return len(self.data["latents_x1"])
    
    def __getitem__(self, idx):
        latent_x1 = torch.load(os.path.join(LATENT_DIR_X1, self.data["latents_x1"][idx]))
        # latent_x2 = torch.load(os.path.join(LATENT_DIR_X2, self.data["latents"][idx]))
        original_caption = self.data["caption"][idx]
        # caption_changed = self.data["caption_2"][idx]
        return {"latent_x1" : latent_x1, "original_caption" : original_caption}

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--use-fsdp", default=False, help="Enable FSDP training or not")
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="disentangle-edit",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=("checkpoint to resume training from"),
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and wandb logging
    if args.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    # if(args.report_to == "wandb"):
    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project=args.tracker_project_name,
    #     )

    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        log_with=args.report_to,
        project_dir=os.path.join(OUTPUT_DIR, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=False,
        kwargs_handlers=[init_handler]
    )

    SEED = init_random_seed(SEED)
    set_random_seed(SEED)

    #creating data-loader
    ds = DisEntangleDataset(DATA_PATH)
    train_loader = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=False)

    #loading models
    model = Mei(input_size = 32, patch_size = 2, in_channels = 4, hidden_size = 768, depth = 12, output_dim=512, num_heads = 16)
    clipModel = CLIPEncoder("cuda")

    model_ema = deepcopy(model).eval()
    ema_update(model_ema, model, 0.)

    optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, eps = 1e-10)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP_STEPS) 

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    tracker_config = {
        "learning_rate" : LEARNING_RATE,
        "latent_size" : 32,
        "training_batch_size": TRAIN_BATCH_SIZE,
        "gradient_clipping": GRADIENT_CLIP,
        "weight_decay": WEIGHT_DECAY,
        "mse_scale": MSE_SCALE
    }

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
    
    start_epoch = 0
    if args.resume_from is not None:
        start_epoch, missing, Unexpected = load_checkpoint(
            args.resume_from, model, model_ema, optimizer, lr_scheduler
        )
    
    # vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH, subfolder = "vae").to(accelerator.device)
    # vae.eval()
    # SCALING_FACTOR = vae.config.scaling_factor
    
    model, model_ema = accelerator.prepare(model, model_ema)
    optimizer, train_loader, lr_scheduler= accelerator.prepare(optimizer, train_loader, lr_scheduler)
    # ods_loss = accelerator.prepare(ods_loss)
    clipModel = accelerator.prepare(clipModel)
    # vae = accelerator.prepare(vae)
    # vae.requires_grad = False
    
    train(model, model_ema, NUM_EPOCHS, train_loader, clipModel, optimizer, lr_scheduler, start_epoch)