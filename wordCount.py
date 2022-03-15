# This script counts all the words currently in dungeons-dataset.csv and outputs the count

with open('dungeons-dataset.csv', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    counts = dict()
    for line in lines:
        # remove new line and punctation
        line = line.strip()
        line = line.replace(".", "")
        line = line.replace(",", "")
        line = line.replace('"', "")
        line = line.split(" ")
        for word in line:
            word = word.lower()
            counts[word] = counts.get(word, 0) + 1

fluff = [
    'a', 'the', 'of', 'and', 'in', 'has', 'to', 'is',
    'this', 'with', 'it', 'are', 'on', 'an', 'as', 'from',
    'at', 'that', 'there', 'by', 'been', 'have', 'be', 'you',
    'one', 'here', 'some'
         ]
for w in sorted(counts, key=counts.get, reverse=True):
    if counts[w] > 10 and w not in fluff:
        print(w, counts[w])