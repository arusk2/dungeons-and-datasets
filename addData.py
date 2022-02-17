import csv

# Function takes text as parameter, returns a list of the sentences that makes up the text.
def filter_text(text):
    text = text.split('. ')
    # add a period if there isn't one.
    for sentence in text:
        if sentence[-1] != '.':
            sentence = sentence + "."
    # remove any empty strings, just in case
    text = [x for x in text if x]
    return text

def import_desc():
    # Testing input for input function
    # Rough input delimited by newline ? since input is paragraphs including . and , symbols
    with open("db_rough.csv", "a") as file:
        writer = csv.writer(file)
        while choice != "-1":
            # can eventually update to add some light metadata about subject of sentence as we parse
            choice = input("Enter a text to add to the database: ")
            choice = filter_text(choice)
            writer.writerow([choice])


def main():
    # import runs until stopped by user.
    import_desc()

if __name__ == "__main__":
        main()