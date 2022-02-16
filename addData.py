import csv

# Testing input for input function
# Rough input delimited by newline ? since input is paragraphs including . and , symbols

choice = input("Enter a text to add to the database: ")
with open("db_rough.csv", "a") as file:
    writer = csv.writer(file)
    while choice != "-1":
        # can eventually update to add some light metadata about subject of sentence as we parse
        writer.writerow([choice])
        choice = input("Enter a text to add to the database: ")
