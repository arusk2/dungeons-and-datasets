import csv

# Function takes text as parameter, returns a list of the sentences that makes up the text.
def filter_text(text):
    text = text.split('. ')
    # add a period if there isn't one.
    # remove empty strings
    text = [x for x in text if x]
    for i in range(len(text)):
        # remove trailing/leading spaces
        text[i] = text[i].strip()
        # remove newline chars
        text[i] = text[i].replace('\n', '')
        if text[i][-1] != ".":
            text[i] = text[i] + "."
    # remove any empty strings, just in case
    return text

def import_desc():
    # Testing input for input function
    # Rough input delimited by newline ? since input is paragraphs including . and , symbols
    choice = ""
    with open("db_rough.csv", "a", encoding='utf-8') as file:
        writer = csv.writer(file)
        while choice != "-1":
            # can eventually update to add some light metadata about subject of sentence as we parse
            choice = input("Enter a text to add to the database: ")
            if choice != "-1":
                choice = filter_text(choice)
                for row in choice:
                    writer.writerow([row])


def main():
    # import runs until stopped by user.
    import_desc()

if __name__ == "__main__":
        main()