import pandas as pd
from tabulate import tabulate

if __name__ == "__main__":
    data = pd.read_pickle('CRD3_spacy_processed_1')
    result = data.query('LocProb > 2.0')
    print(result)
