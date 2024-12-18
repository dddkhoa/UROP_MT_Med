import logging
import os
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import faiss
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset
import nltk
from sentence_transformers import SentenceTransformer

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import torch


LOGGER = logging.getLogger("MINT")

embedding_model = SentenceTransformer("all-mpnet-base-v2")


def build_index_medev():
    if not os.path.exists("/home/ubuntu/21khoa.ddd/urop-mt-med/med_ev_index"):
        os.makedirs("/home/ubuntu/21khoa.ddd/urop-mt-med/med_ev_index")
    
        for split in ["test"]:
            dataset = load_dataset("nhuvo/MedEV", split=split)
            sentences = dataset["text"]
            embeddings = embedding_model.encode(sentences)

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
        
            faiss.write_index(index, f"/home/ubuntu/21khoa.ddd/urop-mt-med/med_ev_index/{split}.index")    

    else:
        print("Index already exists.")


def get_dict_search(dict_file: str, dict_search_inputs: list[str]):
    """
    input:
        dict_search_inputs: list of medical English terms 
    
    return: 
        result: dictionary of medical English terms and their Vietnamese translations
    """
    dict_df = pd.read_excel(dict_file)
    dict_df.fillna("nan", inplace=True)

    englishterms = list(dict_df['English Terms'])
    medict = dict.fromkeys(englishterms)

    for i in range(dict_df.shape[0]):
        medict[dict_df['English Terms'][i]] = dict_df['Vietnamese Terms'][i]

    result = defaultdict(list)

    for input in dict_search_inputs:
        if input in medict:
            if medict[input] != "nan":
                result[input].append(medict[input])

    return result

def get_shots_search(query, index, search_sentences, top_k: int=3):

    query_embedding = embedding_model.encode([query]).astype('float32')
    D, I = index.search(query_embedding, top_k+1)
    try:
        result = [search_sentences[i] for i in I[0]]
    except:
        print("WTF: ", I)
        return []

    result.pop(0) # remove the first element, which is the query itself

    return result


def get_dipmt_prompt(query, num_shots, index, en_search_sentences, vn_search_sentences):
    query_list = set(query.split())
    dict_search = get_dict_search("/home/ubuntu/21khoa.ddd/urop-mt-med/data/Clean_Medict.xlsx", query_list)

    en_few_shot_examples = get_shots_search(query, index, en_search_sentences, num_shots)

    # find equivalent vn_sentences to en_search_sentences in MedEV
    vn_few_shot_examples = []
    for en_sentence in en_few_shot_examples:
        en_sent_index = en_search_sentences.index(en_sentence)
        vn_few_shot_examples.append(vn_search_sentences[en_sent_index])

    if num_shots == 0:
        dipmt_prompt = f"Translate the input sentence to Vietnamese:\n{query}."
        dipmt_prompt += "\nIn this context, "
        for key, value in dict_search.items():
            if len(value) > 1:
                dipmt_prompt += f"The word {key} can mean {', '.join(value)}."
            else:
                dipmt_prompt += f"The word {key} means {value[0]}."

        dipmt_prompt += "\nThe full translation to Vietnamese is: "

    else:
        dipmt_prompt = ""
        for shot in range(num_shots):
            dipmt_prompt += f"\nExample {shot+1}:\n"
            dipmt_prompt += f"Translate the input sentence to Vietnamese:\n{en_few_shot_examples[shot]}."
            dipmt_prompt += "\nIn this context, "
            for key, value in dict_search.items():
                if len(value) > 1:
                    dipmt_prompt += f"The word {key} can mean {', '.join(value)}. "
                else:
                    dipmt_prompt += f"The word {key} means {value[0]}. "
            
            dipmt_prompt += f"\nThe full translation to Vietnamese is: {vn_few_shot_examples[shot]}.\n\n"

        dipmt_prompt += f"Translate the input sentence to Vietnamese:\n{query}."
        dipmt_prompt += "\nIn this context, "
        for key, value in dict_search.items():
            if len(value) > 1:
                dipmt_prompt += f"The word {key} can mean {', '.join(value)}."
            else:
                dipmt_prompt += f"The word {key} means {value[0]}."

        dipmt_prompt += "\nThe full translation to Vietnamese is: "

    return dipmt_prompt


@hydra.main(
    version_base=None, config_path="configs", config_name="llama_3.2"
)
def main(config: DictConfig):
    build_index_medev()
    index = faiss.read_index("/home/ubuntu/21khoa.ddd/urop-mt-med/med_ev_index/test.index")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model.name,
        max_seq_length = config.max_seq_length,
        dtype = config.dtype,
        load_in_4bit = config.load_in_4bit,
    )
    
    dipmt_num_shots = 10

    dataset = load_dataset("nhuvo/MedEV")

    splitindex = len(dataset['test']['text']) // 2
    inp = dataset['test']['text'][0:splitindex]
    outp = dataset['test']['text'][splitindex:]

    splitindex_train = len(dataset['train']['text']) // 2
    inp_train = dataset['train']['text'][0:splitindex_train]
    outp_train = dataset['train']['text'][splitindex_train:]

    FastLanguageModel.for_inference(model)

    with open("dipmt_results.txt", "a") as f:
        allbleusc = []

        f.write(f'Model: {config.model.name}\n')
        f.write(f'Shots max: {dipmt_num_shots}\n')
        f.write(f'Number of testcases: {100}\n')
    
        for i in range(10):
            dipmt_prompt = get_dipmt_prompt(inp[i], dipmt_num_shots, index, inp, outp)

            messages = [
                {"role": "system", "content": "Only output the translated sentence."},
                {"role": "user", "content": dipmt_prompt}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"Input {i}: {dipmt_prompt}")
            print(f"Generated Output {i}: {response}")
            print(f"Expected Output {i}: {outp[i]}")

            bleusc = nltk.translate.bleu_score.sentence_bleu([outp[i].split()], response.split())
            allbleusc.append(bleusc)

        print(f"Average bleu: {sum(allbleusc) / len(allbleusc)}")

if __name__ == "__main__":
    main()
