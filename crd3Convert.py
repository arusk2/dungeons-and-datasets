import os
import json
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
# this script imports the crd3 dataset and converts it from JSON to a continuous, CSV file
# with clear sentence breaks.

path = "CRD3-master/data/aligned data/c=2"
file = "/CRD3_by_sentence.csv"
nlp = English()
nlp.add_pipe('sentencizer')

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        with open(os.path.join(root, name), 'r') as file:
            jsonData = json.load(file)
        print("Datatype of variable: ", type(jsonData))
        # For each of the files, the dialog is stored in the json according to "Turns"
        # Each "Turn" multiple players can talk and so we need to get the "Utterances" from each entry.
        # Then we join them together in a string and write the string out one sentence at a time to our csv file
        for index in jsonData:
            turns = index["TURNS"]
            for turn in turns:
                text = " ".join(turn["UTTERANCES"])
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                print("Here")
