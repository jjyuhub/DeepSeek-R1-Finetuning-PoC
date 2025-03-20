import os
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from datasets import load_dataset

# --- 1. Load secrets and set up environment ---
hf_token = os.getenv("HF_TOKEN")  # Ensure your Hugging Face token is set as an env var
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")

# --- 2. Load and preprocess the dataset ---
# Using a sample medical reasoning dataset from Hugging Face
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[0:500]", trust_remote_code=True)

# Define the EOS token using the tokenizer's value later
# (For now, we use a placeholder; this will be replaced after loading the tokenizer.)
EOS_TOKEN = "<|endoftext|>"

def format_instruction(example):
    # Here we build a prompt that includes a clear instruction and expects a response.
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "Write a response that appropriately completes the request.\n"
        "Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.\n\n"
        "### Instruction:\n{question}\n\n"
        "### Response:\n{answer}\n"
    )
    return {"text": prompt.format(question=example["Question"], answer=example["Response"]) + EOS_TOKEN}

dataset = dataset.map(format_instruction)

# --- 3. Load the model and tokenizer ---
# We use the distilled DeepSeek R1 model (with 8B parameters) optimized for 4-bit quantization.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    token=hf_token,
)

# Update EOS_TOKEN from the actual tokenizer
EOS_TOKEN = tokenizer.eos_token

# --- 4. Prepare the model for PEFT (LoRA) ---
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Optimized for long contexts
    random_state=1000,
    use_rslora=False,
    loftq_config=None,
)

# --- 5. Set up training arguments ---
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=100,  # Adjust this value for longer training runs
    learning_rate=2e-4,
    fp16=not torch.backends.mps.is_available(),  # Use FP16 if MPS (Mac) is not available
    bf16=torch.backends.mps.is_available(),       # Use BF16 on Apple Silicon if available
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=1000,
    output_dir="./outputs",
    report_to="wandb",  # Reports training metrics to Weights & Biases
)

# --- 6. Initialize the trainer ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=training_args,
)

# --- 7. Start training ---
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    # Optionally save the model locally
    trainer.save_model("./finetuned_model")
