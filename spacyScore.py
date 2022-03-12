import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas()


def calc_score(row):
    doc = row
    # doc = row['Text']
    score = 0

    # set up spacy doc objects to compare to
    room = nlp('room')
    floor = nlp('floor')
    door = nlp('door')
    wall = nlp('wall')
    you = nlp('you')

    # parts of speech comparison
    pos_room = nlp('You enter a room')

    # curse words are a sign that this is banter
    boob = nlp('boob')
    dick = nlp('dick')
    fuck = nlp('fuck')
    shit = nlp('shit')

    # if the sentence starts with...
    if doc[0].lemma_ == you[0].lemma_:
        score += 1

    # in one loop go through all the tokens, perform tests to calculate score
    for token in doc:
        # POS comparison
        pos_tokens = []
        if len(pos_tokens) <= 4:
            # store four tokens at a time
            pos_tokens.append(token)
        else:
            if len(pos_tokens) == 4:
                if pos_tokens[0].pos == pos_room[0].pos and pos_tokens[1].pos == pos_room[1].pos and pos_tokens[2].pos \
                        == pos_room[2].pos and pos_tokens[3].pos == pos_room[3].pos:
                    score += 1

        # positive similarity
        if token.similarity(room) > 0.6:
            score += 1
        if token.similarity(floor) > 0.6:
            score += 1
        if token.similarity(door) > 0.6:
            score += 1
        if token.similarity(wall) > 0.6:
            score += 1

        # negative similarity
        if token.similarity(boob) > 0.6:
            score -= 1
        if token.similarity(dick) > 0.6:
            score -= 1
        if token.similarity(fuck) > 0.6:
            score -= 1
        if token.similarity(shit) > 0.6:
            score -= 1
        if token.text == '"':
            score -= 1

    # subtract score for named entities
    score -= len(doc.ents)

    row['LocProb'] = float(score)


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_trf')
    test = nlp('You enter a room, a purple fence is on the left and there\'s boobs')
    # data = pd.read_pickle('./CRD3_spacy_processed_1.gz')
    # first = data.iloc[0]['Text']
    calc_score(test)
    # data = data.progress_apply(calc_score, axis=1)
