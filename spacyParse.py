import spacy
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def spacy_multiprocess(dataframe):
    docs = []
    for doc in tqdm(nlp.pipe(dataframe['Text'].astype('unicode').values, batch_size=1000, n_process=-1),
                    total=len(dataframe['Text'])):
        if doc.has_annotation("DEP"):
            docs.append(doc)
        else:
            docs.append(None)
    return docs


if __name__ == '__main__':
    # load spacy for analysis
    torch.set_num_threads(1)
    nlp = spacy.load('en_core_web_trf')

    # ingest the raw dataset
    data = pd.read_csv('./CRD3_by_sentence.csv')

    # preprocessing, dropping duplicates, reindexing, cutting in half
    data.rename(columns={'Hello everyone.': 'Text'}, inplace=True)
    data.drop_duplicates('Text', inplace=True)
    data.reset_index(inplace=True, drop=True)
    zeros = np.zeros(len(data.index))
    data = data.assign(LocProb=zeros)
    half = int(len(data) / 2)
    data_first = data.iloc[:half, :]
    data_second = data.iloc[half + 1:, :]

    # process the first half
    data_first['SpacyDoc'] = spacy_multiprocess(data_first)
    data_first.to_pickle('./CRD3_spacy_processed_1.gz')

    # process the second half
    data_second['SpacyDoc'] = spacy_multiprocess(data_second)
    data_second.to_pickle('./CRD3_spacy_processed_2.gz')
