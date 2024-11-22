"""
Implementation of program from the paper 'Dictionary-based Phrase-level Prompting of Large Language Models
for Machine Translation'
Ghazvininejad M., Gonen H., Zettlemoyer L.

https://arxiv.org/abs/2302.07856

English-Vietnamese Version
"""

import pandas
from datasets import load_dataset
import torch
import nltk
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from random import randrange
import statistics

# Load model directly
MODELNAME = "Qwen/Qwen2.5-7B-Instruct"

#Setup number of shots
SHOT_N = input("Number of shots (iterated, default = 3): ")
TESTCASES_N = input("Number of testcases (default = 1000): ")

SHOT_N = 3 if not SHOT_N else int(SHOT_N)
TESTCASES_N = 1000 if not TESTCASES_N else int(TESTCASES_N)

# Setting file name
now = datetime.now()
filename = f"{MODELNAME.split("/")[1]}_{now.strftime("%d.%m.%Y_%H.%M.%S")}_responses_zeroshot.txt"

# Model setup
device = "cuda:0" if torch.cuda.is_available() else "cpu" # Option for CUDA GPUs
tokenizer = AutoTokenizer.from_pretrained(MODELNAME)
model = AutoModelForCausalLM.from_pretrained(MODELNAME)
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Reading Excel dictionary
ds = load_dataset("nhuvo/MedEV")
dictexcel = pandas.ExcelFile("data/Sample_Medict.xlsx")
dataframe = pandas.read_excel(dictexcel)

englishterms = list(dataframe['English Terms'])
random_medict = dict.fromkeys(englishterms)

# Transferring data to Python dictionary
for i in range(dataframe.shape[0]):
    random_medict[dataframe['English Terms'][i]] = dataframe['Vietnamese Terms'][i]

# Splitting test bilingual corpus
splitindex = len(ds['test']['text']) // 2
inp = ds['test']['text'][0:splitindex]
outp = ds['test']['text'][splitindex:]
splitindex_train = len(ds['train']['text']) // 2
inp_train = ds['train']['text'][0:splitindex_train]
outp_train = ds['train']['text'][splitindex_train:]

with open(filename, "a") as f:
    f.write(f'Model: {MODELNAME}\n')
    f.write(f'Shots max: {SHOT_N}\n')
    f.write(f'Number of testcases: {TESTCASES_N}\n')

# Prompting and response generation
for shot in range(SHOT_N):
    allbleusc = []

    with open(filename, "a") as f:
        f.write(f'****************************\n')
        f.write(f'Shots: {shot}\n')
        f.write(f'****************************\n')

    for i in range(TESTCASES_N):
        # Prompt parsing + dictionary meaning embedding
        sentence = inp[i]
        prompt = ""

        for j in range(shot):
            randsentence = randrange(0, splitindex_train)
            trainsentence = inp_train[randsentence]
            traintranslation = outp_train[randsentence]

            definitions = []
            for elem in list(random_medict.keys()):
                if elem in trainsentence:
                    definitions.append(f"the word {elem} means {random_medict[elem]}")

            prompt = prompt + (f"""Translate the following sentence to Vietnamese:\n{trainsentence}{"." if trainsentence[-1] != "." else ""}
{f"In this context, {"; ".join(definitions)}\n" if definitions else ""}The full translation to Vietnamese is: {traintranslation}\n\n""")

        definitions = []
        for elem in list(random_medict.keys()):
            if elem in sentence:
                definitions.append(f"the word {elem} means {random_medict[elem]}")

        prompt = prompt + (f"""Translate the following sentence to Vietnamese: {sentence}{"." if sentence[-1] != "." else ""}
{f"In this context, {"; ".join(definitions)}\n" if definitions else ""}The full translation to Vietnamese is:""")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model = model.to(device)

        generate_ids = model.generate(inputs.input_ids, attention_mask=inputs["attention_mask"], pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(prompt) + 1:]

        #responses.append(response)

        bleusc = nltk.translate.bleu_score.sentence_bleu([outp[i].split()], response.split())
        allbleusc.append(bleusc)

        with open(filename, "a") as f:
            f.write(f"------Response {i + 1}------\n\n")
            f.write(f"Prompt:\n{prompt}\n\nExpected response:\n{outp[i]}\n\nLLM-generated response:\n")
            f.write(response + "\n\n")
            f.write(f"BLEU Score: {bleusc}\n\n")

        print(f"{i} responses generated")

    with open(filename, "a") as f:
        f.write(f'BLEU scores statistics for #{shot}-shot translation')
        f.write(f'Mean: {statistics.mean(allbleusc)}\n')
        f.write(f'Median: {statistics.median(allbleusc)}\n')
        f.write(f'Standard Deviation: {statistics.stdev(allbleusc)}\n')
        f.write(f'Variance: {statistics.variance(allbleusc)}\n\n\n')

f.close()
# print(*responses, sep="\n")
