import train
dataset = train.read_data()
train, test = train.prep_train_test_split(dataset)
model, tokenizer = train.train_model(train, test)