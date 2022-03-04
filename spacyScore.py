import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas()


def calc_score(row):
    doc = row['Text']
    score = 0
    room = nlp('room')
    floor = nlp('floor')
    door = nlp('door')
    wall = nlp('wall')
    # in one loop go through all the tokens, perform tests to calculate score
    for token in doc:
        if token.lemma_.similarity(room) > 0.8:
            score += 1
        if token.lemma_.similarity(floor) > 0.8:
            score += 1
        if token.lemma_.similarity(door) > 0.8:
            score += 1
        if token.lemma_.similarity(wall) > 0.8:
            score += 1


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_trf')

    data = pd.read_pickle('./CRD3_spacy_processed_1.gz')
    first = data.iloc[0]['Text']
    calc_score(first)
    # data = data.progress_apply(calc_score, axis=1)
