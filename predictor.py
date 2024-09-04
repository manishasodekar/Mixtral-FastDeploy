import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json

# Configuration for 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Use NormalFloat4 (NF4) quantization
    bnb_4bit_compute_dtype=torch.float16  # Use FP16 for computations
)

# Load model with Flash Attention, 4-bit quantization, and tokenizer
# model_name = "mistralai/Mixtral-8x7B-v0.1"
model_name = "ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model with quantization and Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_4bit=True,
    # quantization_config=True,
    torch_dtype=torch.float16,
    # attn_implementation="flash_attention_2",  # Enable Flash Attention v2
    device_map="auto"
)
# .to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Prediction function
def generate_text(prompt, max_new_tokens=100, do_sample=True):
    print(f"Generating text for prompt: {prompt}")
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()


# Batch processing for multiple prompts
def predictor(input_data, batch_size=8, max_new_tokens=100):
    results = []
    prompts = [entry["prompt"] for entry in input_data]  # Extract prompts from input data

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_results = [generate_text(prompt, max_new_tokens=max_new_tokens) for prompt in batch_prompts]
        results.extend(batch_results)

    return results


if __name__ == "__main__":
    # Warmup
    warmup_prompt = "This is a test prompt"
    generate_text(warmup_prompt)

    # Getting input data from the command line
    import sys

    input_data_json = sys.argv[1]  # Read the JSON input passed via command line
    input_data = json.loads(input_data_json)  # Parse the JSON input

    start_time = time.time()
    result = predictor(input_data, max_new_tokens=100)
    print(f"Time taken: {time.time() - start_time} seconds")
    print("Generated Results: \n", result)
