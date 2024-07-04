import os
import glob
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# Configuration
gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
base_directory = f"/mnt/pub1/laion/parquets/data_gpu{gpu_id}"
output_suffix = "_updated.parquet"  # Suffix for processed files
model_id = "jan-hq/Mistral-7B-Instruct-v0.2-SLERP"
batch_size = 64  # Adjust this to fit your GPU memory

# Set the specific GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Assume only one Parquet file per GPU directory
file_path = os.path.join(base_directory, "combined_parquet.parquet")
output_file_path = file_path.replace(".parquet", output_suffix)

if os.path.exists(output_file_path):
    print(f"Skipping already processed file: {output_file_path}")
else:
    df = pd.read_parquet(file_path, engine='pyarrow')
    df["caption_2"] = ''
    df_start_ind = df.index[0]
    prompt = """Given the old image caption: "{}", generate a new caption changing only one image or background features such as object color(specific - do not include colourful or rainbow) or shape or pattern, add or remove new object, changing artistic style, or any other creative changes you would like to see in the image (please limit the change to one of image or background). please limit the caption to 15 words and do not deviate from original prompt."""

    # Process in batches
    for start in tqdm(range(0, len(df), batch_size)):
        end = start + batch_size
        batch = df.iloc[start:end]
        queries = [prompt.format(caption) for caption in batch["caption"]]

        responses = pipe(
            queries,
            do_sample=True,
            max_new_tokens=40,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            batch_size=min(batch_size, len(batch))  # Adjust batch size for the last batch
        )

        for i, result in enumerate(responses):
            # Ensure the result is correctly formatted
            if 'generated_text' in result[0].keys():
                response = result[0]['generated_text'][len(queries[i]):].strip().replace("Updated Image Caption: ", "").replace("&nbsp;.", "").replace("Modified Image Caption: ", "").split(".")[0].split("\n")[0] + "."
            else:
                response = "No valid response generated."
            df.at[df_start_ind + start + i, 'caption_2'] = response
            
        if start % 1000 == 0:  # Adjust based on your preference
            intermediate_save_path = f"/mnt/pub1/laion/captioned_parquets/data_gpu{gpu_id}/intermediate_{start}.parquet"
            df[:start].to_parquet(intermediate_save_path)
            print(f"Saved intermediate results to {intermediate_save_path}")

    # Save the updated DataFrame
    df.to_parquet(output_file_path)
    print(f"Processed and saved data to {output_file_path}")

print("Processing complete.")
