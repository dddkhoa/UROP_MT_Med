import logging
import os
import random
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset
import nltk

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import torch


LOGGER = logging.getLogger("MINT")


@hydra.main(
    version_base=None, config_path="configs", config_name="qwen_2.5"
)
def main(config: DictConfig):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name, 
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        token="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",
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.model.rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = True, # True or "unsloth" for very long context
        random_state = 47,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = os.path.join(config.outputs_dir, "lora_model"),
    #     max_seq_length = config.max_seq_length,
    #     dtype = config.dtype,
    #     load_in_4bit = config.load_in_4bit,
    # )

    alpaca_prompt = """Translate the input sentence to Vietnamese.
    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for input, output in zip(inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("nhuvo/MedEV")

    splitindex = len(dataset['test']['text']) // 2
    inp = dataset['test']['text'][0:splitindex]
    outp = dataset['test']['text'][splitindex:]

    splitindex_train = len(dataset['train']['text']) // 2
    inp_train = dataset['train']['text'][0:splitindex_train]
    outp_train = dataset['train']['text'][splitindex_train:]

    dataset = Dataset.from_dict({"input": inp_train, "output": outp_train})
    dataset = dataset.map(formatting_prompts_func, batched = True, num_proc=4)

    LOGGER.info(f"Sample data: {dataset['text'][0]}")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = config.max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 16,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, 
            max_steps = 60,
            learning_rate = 3e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 47,
            output_dir = config.outputs_dir,
            report_to = "none", # wandb
        )
    )

    trainer_stats = trainer.train()

    LOGGER.info(f"Trainer stats: {trainer_stats}")

    model.save_pretrained(os.path.join(config.outputs_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(config.outputs_dir, "lora_model"))

    FastLanguageModel.for_inference(model)

    allbleusc = []
    for i in range(100):
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                # "Translate the input sentence to Vietnamese.", # instruction
                inp[i], # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        start_response_idx = response.find("### Response:\n") + len("### Response:\n")
        response = response[start_response_idx+1:].strip()

        bleusc = nltk.translate.bleu_score.sentence_bleu([outp[i].split()], response.split())
        allbleusc.append(bleusc)

    LOGGER.info(f"Average bleu: {sum(allbleusc) / len(allbleusc)}")

if __name__ == "__main__":
    main()
