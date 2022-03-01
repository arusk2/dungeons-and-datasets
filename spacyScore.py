import pandas as pd
from spacy import displacy

if __name__ == '__main__':
    data = pd.read_pickle('./CRD3_spacy_processed_1.gz')
    doc = data.iloc[0]['Text']
    displacy.serve(doc, style="ent")
