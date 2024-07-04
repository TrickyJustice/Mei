import pandas as pd
import torch
from PIL import Image
from diffusers import AutoencoderKL
import torchvision.transforms as transforms 
import os

#CONFIG
MODEL_PATH = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
IMAGE_EXTENSION = ".jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the Parquet file
df = pd.read_parquet('/mnt/pub1/laion/data/final_parquet.parquet', engine = "pyarrow")


vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder = "vae").to(device)
vae.eval()

SCALING_FACTOR = vae.config.scaling_factor

transform = transforms.Compose([ 
    transforms.PILToTensor(),
]) 

def encode_img(input_img):
    # Single image -> single latent in a batch (so size 1, 4, 32, 32)
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img*2 - 1)
    return latent.latent_dist.sample() * SCALING_FACTOR

latent_dir_x1 = "latents_directory_x1_one_mil"
os.makedirs(latent_dir_x1, exist_ok=True)

df['latents_x1'] = ''  # Add an empty 'latents' column to the DataFrame

# Iterate over rows in the dataframe
for index, row in df.iterrows():
    image_key = row['key']
    image_dir_path = "/mnt/pub1/laion/data/all_images"
    # Generate latent
    try:
        image_name = image_key + IMAGE_EXTENSION
        image_path = os.path.join(image_dir_path, image_name)
        img = transform(Image.open(image_path).convert("RGB")).to(torch.float32).to(device)/255
        # print(img)

        latent = encode_img(img).detach()
        # print(latent.shape)
        
        # Save the latent tensor
        latent_filename = f"{image_key}_latent.pt"
        file_path = os.path.join(latent_dir_x1, latent_filename)
        torch.save(latent, file_path)
        
        df.at[index, 'latents_x1'] = latent_filename
        
        print(f"Saved latent for {image_key} successfully.")
    except Exception as e:
        print(f"Error generating or saving latent for {image_key}: {e}")

df.to_parquet('Mistral_captions_with_latents_one_mil.parquet')

print("Updated Parquet file has been saved with latent filenames.")