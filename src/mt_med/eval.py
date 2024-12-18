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
        model_name = config.model.name,
        max_seq_length = config.max_seq_length,
        dtype = config.dtype,
        load_in_4bit = config.load_in_4bit,
    )

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

    print(dataset['text'][0])

    FastLanguageModel.for_inference(model)

    allbleusc = []
    for i in range(100):
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                inp[i], # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        start_response_idx = response.find("### Response:\n") + len("### Response:\n")
        response = response[start_response_idx+1:].strip()

        print(f"Input {i}: {inp[i]}")
        print(f"Output {i}: {response}")

        bleusc = nltk.translate.bleu_score.sentence_bleu([outp[i].split()], response.split())
        allbleusc.append(bleusc)

    print(f"Average bleu: {sum(allbleusc) / len(allbleusc)}")

if __name__ == "__main__":
    main()
