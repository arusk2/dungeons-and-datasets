import spacy
import pandas as pd
import numpy as np
from pandasgui import show

if __name__ == '__main__':
    # ingest the raw dataset
    data = pd.read_csv('./CRD3_by_sentence.csv')
    data.rename(columns={'Hello everyone.': 'Text'}, inplace=True)
    zeros = np.zeros(len(data.index))
    data = data.assign(LocProbability=zeros)
    show(data)

    # load spacy for analysis
    nlp = spacy.load('en_core_web_trf')
