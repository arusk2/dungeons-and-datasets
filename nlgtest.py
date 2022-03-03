# Testing fine tune & generation with cur dataset
from datetime import datetime
import pandas as pd
import datasets
import transformers
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import seaborn as sns
from datasets import Dataset, load_dataset

# import models
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>')
model = TFGPT2LMHeadModel.from_pretrained('gpt2', use_cache=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)

# read in data
data = pd.read_csv('db_rough.csv')
data.columns = ["text"]
# convert to dataset
dataset = Dataset.from_pandas(data)
# preprocess and set up tokenizer
# Following https://data-dive.com/finetune-german-gpt2-on-tpu-transformers-tensorflow-for-text-generation-of-reviews
# They mention that extra preprocessing needs to be done for TF version of GPT-2 so we'll use their guide
MAX_TOKENS = 128
def tokenize_fn(examples, tokenizer=tokenizer):
    examples = [ex + "<|endoftext|>" for ex in examples["text"]] # add EOS token manually.
    output = tokenizer(examples, add_special_tokens=True, max_length=MAX_TOKENS, truncation=True,
                       pad_to_max_length=True)
    # Apparently the labels aren't auto-shifted in TF models. In order to train, GPT needs to predict
    # next word. This is done by token id and needs to be manually shifted for now.
    # Using method from link above
    output["labels"] = [x[1:] for x in output["input_ids"]]
    output["labels"] = [
        [-100 if x == tokenizer.pad_token_id else x for x in y]
        for y in output["labels"]
    ]
    # truncate input ids and attention mask to account for label shift
    output["input_ids"] = [x[:-1] for x in output["input_ids"]]
    output["attention_mask"] = [x[:-1] for x in output["attention_mask"]]
    return output

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=True)
# test set is 10% train for now since we don't have a lot of data yet
dataset = dataset.train_test_split(test_size=.1, shuffle=True, load_from_cache_file=True)

# Convert to Tensors for TF
train_tensor_inputs = tf.convert_to_tensor(dataset["train"]["input_ids"])
train_tensor_labels = tf.convert_to_tensor(dataset["train"]["labels"])
train_tensor_mask = tf.convert_to_tensor(dataset["train"]["attention_mask"])
train = tf.data.Dataset.from_tensor_slices(
    ({"input_ids": train_tensor_inputs, "attention_mask": train_tensor_mask},
        train_tensor_labels,)
)
test_tensor_inputs = tf.convert_to_tensor(dataset["test"]["input_ids"])
test_tensor_labels = tf.convert_to_tensor(dataset["test"]["labels"])
test_tensor_mask = tf.convert_to_tensor(dataset["test"]["attention_mask"])
test = tf.data.Dataset.from_tensor_slices(
    ({"input_ids": test_tensor_inputs, "attention_mask": test_tensor_mask},
        test_tensor_labels,)
)

# Now to train the model
epochs = 5
batch_size = 4
init_learning_rate = .001
buffer_size = len(train)
train_ds = (
    train.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
)
test_ds = test.batch(batch_size, drop_remainder=True)

# Schedule LR to decay as training goes
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    init_learning_rate,
    decay_steps=500,
    decay_rate=0.7,
    staircase=True)
model.resize_token_embeddings(len(tokenizer))
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=model.compute_loss)
model.summary()

# Create Callbacks
now = datetime.now().strftime("%Y-%m-%d_%H%M")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", verbose=1, patience=1, restore_best_weights=True
    )
]
""" Checkpointing was weird so I removed it from the callbacks. Will troubleshoot later.
    tf.keras.callbacks.ModelCheckpoint(
        "/data/models/" + now + "_GPT2-Model_{epoch:02d}_{val_loss:.4f}.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
"""
# Get ready for training.
steps_per_epoch = int(buffer_size // batch_size)
print(
    f"Model Params:\nbatch_size: {batch_size}\nEpochs: {epochs}\n"
    f"Step p. Epoch: {steps_per_epoch}\n"
    f"Initial Learning rate: {init_learning_rate}"
)
hist = model.fit(
    train_ds,
    validation_data=test_ds,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)

# Chart Loss
loss = pd.DataFrame(
    {"train loss": hist.history["loss"], "test loss": hist.history["val_loss"]}
).melt()
loss["epoch"] = loss.groupby("variable").cumcount() + 1
sns.lineplot(x="epoch", y="value", hue="variable", data=loss).set(
    title="Model loss",
    ylabel="",
    xticks=range(1, loss["epoch"].max() + 1),
    xticklabels=loss["epoch"].unique(),
)

## Now that training is done lets generate some sentences

from transformers import pipeline

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
gen_data = gen("You enter", max_length=512, num_return_sequences=10)
gen_data = pd.DataFrame(gen_data)
for sentence in gen_data["generated_text"]:
    print(sentence)
