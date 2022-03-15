import pandas as pd
from tabulate import tabulate

if __name__ == "__main__":
    data = pd.read_csv('dungeons-dataset.csv', sep='|', names=["LocationDescriptions"])
    with open('dungeons-dataset.html', 'w') as outputfile:
        outputfile.write(tabulate(data, headers="keys", tablefmt="html"))
    data.to_pickle('dungeons-dataset.pkl')
