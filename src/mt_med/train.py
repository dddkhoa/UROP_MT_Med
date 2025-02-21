# Import necessary libraries
import logging
import os
import random
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset
import nltk

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Set up logging
LOGGER = logging.getLogger("MINT")

# Main training pipeline using Hydra for configuration management
@hydra.main(
    version_base=None, config_path="configs", config_name="qwen_2.5"
)
def main(config: DictConfig):
    # ===== 1. Model Initialization =====
    # Load the base language model with specified configurations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name, 
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        token="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",
    )
    
    # ===== 2. Model Fine-tuning Setup =====
    # Configure LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.model.rank,  # LoRA rank for parameter efficiency
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 47,
        use_rslora = False,
        loftq_config = None,
    )

    # ===== 3. Data Formatting =====
    # Define the prompt template for English to Vietnamese translation
    alpaca_prompt = """Translate the input sentence to Vietnamese.
    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token

    # Function to format examples into prompt-completion pairs
    def formatting_prompts_func(examples):
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for input, output in zip(inputs, outputs):
            text = alpaca_prompt.format(input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    # ===== 4. Data Loading =====
    # Load training, validation, and test data from files
    with open(f"{config.data_dir}/train.en.txt", "r", encoding="utf-8") as f_en:
        train_en = [line.strip() for line in f_en.readlines()]
    
    with open(f"{config.data_dir}/train.vi.txt", "r", encoding="utf-8") as f_vi:
        train_vi = [line.strip() for line in f_vi.readlines()]

    # Read the validation files
    with open(f"{config.data_dir}/val.en.txt", "r", encoding="utf-8") as f_en:
        val_en = [line.strip() for line in f_en.readlines()]
    
    with open(f"{config.data_dir}/val.vi.txt", "r", encoding="utf-8") as f_vi:
        val_vi = [line.strip() for line in f_vi.readlines()]

    # Read the test files
    with open(f"{config.data_dir}/test.en.txt", "r", encoding="utf-8") as f_en:
        test_en = [line.strip() for line in f_en.readlines()]
    
    with open(f"{config.data_dir}/test.vi.txt", "r", encoding="utf-8") as f_vi:
        test_vi = [line.strip() for line in f_vi.readlines()]

    # ===== 5. Dataset Creation =====
    # Create HuggingFace datasets for training and validation
    train_dataset = Dataset.from_dict({
        "input": train_en,
        "output": train_vi
    })
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=4)

    val_dataset = Dataset.from_dict({
        "input": val_en,
        "output": val_vi
    })
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True, num_proc=4)

    # Log sample data for verification
    LOGGER.info(f"Sample training data: {train_dataset['text'][0]}")
    LOGGER.info(f"Sample validation data: {val_dataset['text'][0]}")

    # ===== 6. Training Configuration =====
    # Initialize SFT (Supervised Fine-Tuning) trainer with training parameters
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = config.max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            # Training hyperparameters
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, 
            max_steps = 60,
            learning_rate = 3e-4,
            # Hardware optimization settings
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            # Logging and evaluation settings
            logging_steps = 1,
            eval_steps = 10,
            evaluation_strategy = "steps",
            # Optimization settings
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 47,
            output_dir = config.outputs_dir,
            report_to = "none",
        )
    )

    # ===== 7. Model Training =====
    # Execute the training process
    trainer_stats = trainer.train()
    LOGGER.info(f"Trainer stats: {trainer_stats}")

    # ===== 8. Model Saving =====
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(os.path.join(config.outputs_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(config.outputs_dir, "lora_model"))

    # Prepare model for inference
    FastLanguageModel.for_inference(model)

    # ===== 9. Model Evaluation =====
    # Evaluate the model on test set using BLEU score
    allbleusc = []
    for i in range(len(test_en)):
        # Generate translations for test inputs
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                test_en[i], # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        # Generate translation
        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Extract the generated translation
        start_response_idx = response.find("### Response:\n") + len("### Response:\n")
        response = response[start_response_idx+1:].strip()

        # Calculate BLEU score for each translation
        bleusc = nltk.translate.bleu_score.sentence_bleu([test_vi[i].split()], response.split())
        allbleusc.append(bleusc)

    # Log final evaluation results
    LOGGER.info(f"Average bleu: {sum(allbleusc) / len(allbleusc)}")

if __name__ == "__main__":
    main()
