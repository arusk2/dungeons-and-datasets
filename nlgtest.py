# Testing fine tune & generation with cur dataset
import csv
from transformers import GPT2Tokenizer, TFGPT2Model

# import models
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = TFGPT2Model.from_pretrained('gpt2-medium')

with open('db_rough.csv', 'r') as file:
    csv_reader = csv.reader(file)
    data = list(csv_reader)

batch_tokened = tokenizer(data, padding=True, truncation=True, return_tensors="tf")
