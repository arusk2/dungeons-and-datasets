import spacy
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pandasgui import show
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def calc_score(doc, model):
    score = 0

    # set up spacy doc objects to compare to
    room = model('room')
    floor = model('floor')
    door = model('door')
    wall = model('wall')
    you = model('you')

    # parts of speech comparison
    pos_room = model('You enter a room')

    # curse words are a sign that this is banter
    boob = model('boob')
    dick = model('dick')
    fuck = model('fuck')
    shit = model('shit')

    # if the sentence starts with...
    if doc[0].lemma_ == you[0].lemma_:
        score += 1

    # in one loop go through all the tokens, perform tests to calculate score
    pos_tokens = []
    for token in doc:
        # POS comparison
        if len(pos_tokens) <= 4:
            # store four tokens at a time
            pos_tokens.append(token)
        else:
            if len(pos_tokens) == 2:
                if pos_tokens[0].pos == pos_room[0].pos and pos_tokens[1].pos == pos_room[1].pos:
                    score += 1
            if len(pos_tokens) == 4:
                if pos_tokens[0].pos == pos_room[0].pos and pos_tokens[1].pos == pos_room[1].pos and pos_tokens[2].pos \
                        == pos_room[2].pos and pos_tokens[3].pos == pos_room[3].pos:
                    score += 1
            pos_tokens.clear()

        # positive similarity
        if token.has_vector:
            if token.similarity(room) > 0.5:
                score += 1
            if token.similarity(floor) > 0.5:
                score += 1
            if token.similarity(door) > 0.5:
                score += 1
            if token.similarity(wall) > 0.5:
                score += 1

            # negative similarity
            if token.similarity(boob) > 0.5:
                score -= 1
            if token.similarity(dick) > 0.5:
                score -= 1
            if token.similarity(fuck) > 0.5:
                score -= 1
            if token.similarity(shit) > 0.5:
                score -= 1
            if token.text == '"':
                score -= 2

    # subtract score for named entities
    score -= len(doc.ents)

    return float(score)


def spacy_multiprocess(dataframe):
    # shallow copy the dataframe for results
    result = pd.DataFrame(columns=dataframe.columns)
    # run an nlp pipe over the dataframe, multiprocessing here if capable
    for doc in tqdm(nlp.pipe(dataframe['Text'].astype('unicode').values, batch_size=500, n_process=1),
                    total=len(dataframe['Text'])):
        # if the doc has been processed successfully
        if doc.has_annotation("DEP"):
            # calculate the score of the room
            room_prob = calc_score(doc, nlp)
            if room_prob >= 1 and len(doc) > 5:
                # append it to the result dataframe
                row = pd.Series([doc, room_prob], index=dataframe.columns)
                result = result.append(row, ignore_index=True)
    return result


if __name__ == '__main__':
    # apparently you have to do this or torch hangs forever
    torch.set_num_threads(1)
    # load spacy for analysis
    nlp = spacy.load('en_core_web_lg')

    # ingest the raw dataset
    data = pd.read_csv('./CRD3_by_sentence.csv')

    # preprocessing, dropping duplicates, reindexing, cutting in half
    data.rename(columns={'Hello everyone.': 'Text'}, inplace=True)
    data.drop_duplicates('Text', inplace=True)
    data.reset_index(inplace=True, drop=True)
    zeros = np.zeros(len(data.index))
    data = data.assign(LocProb=zeros)
    half = int(len(data) / 2)
    data_first = data.iloc[:1000, :]
    data_second = data.iloc[half + 1:, :]

    # process the first half
    data_first = spacy_multiprocess(data_first)
    show(data_first)
    data_first.to_pickle('./CRD3_spacy_processed_1')

    # process the second half
    # data_second = spacy_multiprocess(data_second)
    # data_second.to_pickle('./CRD3_spacy_processed_2')


