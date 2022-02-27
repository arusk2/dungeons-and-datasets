import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()


def location_prob(row):
    doc = nlp(row['Text'])
    row['Text'] = doc
    return row


if __name__ == '__main__':
    # load spacy for analysis
    nlp = spacy.load('en_core_web_trf')

    # ingest the raw dataset
    data = pd.read_csv('./CRD3_by_sentence.csv')
    data.rename(columns={'Hello everyone.': 'Text'}, inplace=True)
    zeros = np.zeros(len(data.index))
    data = data.assign(LocProb=zeros)

    data.progress_apply(location_prob, axis=1)
    data.to_pickle('./CRD3_spacy_processed.gz')
