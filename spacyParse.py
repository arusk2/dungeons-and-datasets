import spacy
import pandas as pd
import numpy as np
import pandasgui as pdgui
from tqdm import tqdm

tqdm.pandas()
to_drop = []


def location_prob(row):
    if isinstance(row['Text'], str):
        doc = nlp(row['Text'])
        row['Text'] = doc
    else:
        to_drop.append(row.name)
    return row


if __name__ == '__main__':
    # load spacy for analysis
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
    data_second = data.iloc[half+1:, :]

    # process the first half
    data_first = data_first.progress_apply(location_prob, axis=1)
    data_first.drop(to_drop, inplace=True)
    data_first.reset_index(inplace=True, drop=True)
    data_first.to_pickle('./CRD3_spacy_processed_1.gz')

    # process the second half
    to_drop.clear()
    data_second = data_second.progress_apply(location_prob, axis=1)
    data_second.drop(to_drop, inplace=True)
    data_second.reset_index(inplace=True, drop=True)
    data_second.to_pickle('./CRD3_spacy_processed_2.gz')



