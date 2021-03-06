import csv
import os
import json
import spacy
import re
import pandas as pd

nlp = spacy.load('en_core_web_sm')
# this script imports the crd3 dataset and converts it from JSON to a continuous, CSV file
# with clear sentence breaks.

path = "CRD3-master/data/aligned data/c=2"
write_to_file = "CRD3_by_sentence.csv"
nlp.add_pipe('sentencizer')

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        print(f"Reading in file: {root}/{name}")
        with open(os.path.join(root, name), 'r', encoding='utf-8') as file:
            jsonData = json.load(file)
        # For each of the files, the dialog is stored in the json according to "Turns"
        # Each "Turn" multiple players can talk and so we need to get the "Utterances" from each entry.
        # Then we join them together in a string and write the string out one sentence at a time to our csv file
        for index in jsonData:
            turns = index["TURNS"]
            for turn in turns:
                text = " ".join(turn["UTTERANCES"])
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                with open(write_to_file, 'a', newline='') as file_to_write:
                    writer = csv.writer(file_to_write)
                    # for whatever reason, some sentences have utf-8 characters. These will be removed w/ reg ex
                    for sentence in sentences:
                        sentence = re.sub(r'[^\x00-\x7f]', "", sentence)
                        writer.writerow([sentence])
