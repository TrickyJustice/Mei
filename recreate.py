# import pandas as pd
# import torch
# from diffusers import StableDiffusionPipeline
# import os
# import torch
# from diffusers import Transformer2DModel
# from PixArt_sigma.scripts.diffusers_patches import pixart_sigma_init_patched_inputs, PixArtSigmaPipeline

# # Load the Parquet file
# df = pd.read_parquet('/root/achint/InvertEdit/Mistral_generated_captions_20000-500000.parquet', engine = "pyarrow")


# assert getattr(Transformer2DModel, '_init_patched_inputs', False), "Need to Upgrade diffusers: pip install git+https://github.com/huggingface/diffusers"
# setattr(Transformer2DModel, '_init_patched_inputs', pixart_sigma_init_patched_inputs)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# weight_dtype = torch.float16

# transformer = Transformer2DModel.from_pretrained(
#     "PixArt-alpha/PixArt-Sigma-XL-2-256x256", 
#     subfolder='transformer', 
#     torch_dtype=weight_dtype,
#     use_safetensors=True,
# )
# pipe = PixArtSigmaPipeline.from_pretrained(
#     "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
#     transformer=transformer,
#     torch_dtype=weight_dtyped,
#     use_safetensors=True,
# )
# pipe.to(device)

# # from diffusers import PixArtAlphaPipeline

# # pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-256x256", subdirectory = "transformer",torch_dtype=torch.float16)
# # pipe = pipe.to("cuda")

# # Initialize the model
# # model_id = "runwayml/stable-diffusion-v1-5"
# # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# # pipe = pipe.to("cuda")

# # Directory to save latents
# latent_dir = "latents_directory"
# os.makedirs(latent_dir, exist_ok=True)

# df['latents'] = ''  # Add an empty 'latents' column to the DataFrame

# # Iterate over rows in the dataframe
# for index, row in df.iterrows():
#     prompt = row['caption_2']
#     image_key = row['key']
    
#     # Generate latent
#     try:
#         latent = pipe(prompt, height=256, width=256, output_type='latent').images[0]
        
#         # Save the latent tensodr
#         latent_filename = f"{image_key}_latent.pt"
#         file_path = os.path.join(latent_dir, latent_filename)
#         torch.save(latent, file_path)
#         print(latent.shape)
        
#         df.at[index, 'latents'] = latent_filename
        
#         print(f"Saved latent for {image_key} successfully.")
#     except Exception as e:
#         print(f"Error generating or saving latent for {image_key}: {e}")

# df.to_parquet('mistral_latents.parquet')

# print("Updated Parquet file has been saved with latent filenames.")


import pandas as pd
import torch
from diffusers import Transformer2DModel
from PixArt_sigma.scripts.diffusers_patches import pixart_sigma_init_patched_inputs, PixArtSigmaPipeline
import os

# Load the Parquet file
df = pd.read_parquet('/root/achint/InvertEdit/Mistral_captions_and_latents_x1.parquet', engine="pyarrow")

assert hasattr(Transformer2DModel, '_init_patched_inputs'), "Need to Upgrade diffusers: pip install git+https://github.com/huggingface/diffusers"
setattr(Transformer2DModel, '_init_patched_inputs', pixart_sigma_init_patched_inputs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

transformer = Transformer2DModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
    subfolder='transformer',
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=transformer,
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)

latent_dir = "latents_directory_x2"
os.makedirs(latent_dir, exist_ok=True)

df['latents'] = ''

batch_size = 16  # Set the batch size as appropriate for your GPU
num_batches = (len(df) + batch_size - 1) // batch_size

for i in range(num_batches):
    batch_df = df.iloc[i*batch_size:(i+1)*batch_size]
    prompts = batch_df['caption_2'].tolist()
    
    try:
        latents = pipe(prompts, height=256, width=256, output_type='latent').images
        
        for j, latent in enumerate(latents):
            row_index = i*batch_size + j
            image_key = df.at[row_index, 'key']
            latent_filename = f"{image_key}_latent.pt"
            file_path = os.path.join(latent_dir, latent_filename)
            torch.save(latent, file_path)
            df.at[row_index, 'latents'] = latent_filename
            print(f"Saved latent for {image_key} successfully.")
            
    except Exception as e:
        print(f"Error in batch {i}: {e}")

df.to_parquet('mistral_latents_and_captions.parquet')
print("Updated Parquet file has been saved with latent filenames.")
