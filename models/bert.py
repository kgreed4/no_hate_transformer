from datasets import load_dataset,DatasetDict
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# We will create a small function that will take care of tokenization and encoding
def encode_texts(tokenizer, texts, max_length):
    encoding = tokenizer.batch_encode_plus(
        texts,
        # This is required to add special tokens such as the [CLS] and [SEP] tokens that indicate the start and end of a sentence
        add_special_tokens=True,
        # Here the padding variable is responsible for padding the sequences to the same length
        padding='max_length',
        # The max length of the tokenized sequences
        max_length=max_length,
        return_attention_mask=True,
        # Here we specify that we want the output to be TensorFlow tensors
        return_tensors='tf',
        # If the sequence is longer than max_length, it will be truncated to a fixed length
        truncation=True
    )
    # The encoding['input_ids'] contains the tokenized sequences
    # The encoding['attention_mask'] contains the attention masks and tells the model which tokens to pay attention to and which ones to ignore (mask token)
    return encoding['input_ids'], encoding['attention_mask']

def main():
    # Define datapath
    DATA_PATH = "../data/cleaned_data_nosw.csv"

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Convert to string
    df['tweet'] = str(df['tweet'])

    # Drop every column that isn't tweet or class
    df = df.drop(df.columns.difference(['tweet', 'class']), axis=1)

    # First, we want to use the tokenizer to tokenize and encode the dataset into embeddings
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Here we define the maximum length (randomly chosen per ChatGPT's recommendation)
    max_length = 128

    # We can then call the function to tokenize and encode the dataset
    input_ids, attention_masks = encode_texts(tokenizer, df['tweet'].tolist(), max_length)

    # Here we create labels from the 'class' column
    # This is the target variable that we want to predict
    labels_np = tf.convert_to_tensor(df['class'].values, dtype=tf.int32).numpy()

    # For some reason, I was getting an error saying that I needed to convert to NumPy arrays instead of TensorFlow tensors
    # So I converted the input_ids and attention_masks to NumPy arrays
    input_ids_np = input_ids.numpy()
    attention_masks_np = attention_masks.numpy()

    # Here we split the data into training, validation, and test sets
    train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(input_ids_np, labels_np, random_state=2021, test_size=0.1)
    train_val_masks, test_masks, _, _ = train_test_split(attention_masks_np, labels_np, random_state=2021, test_size=0.1)

    # Here we further split the training set into training and validation sets
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_val_inputs, train_val_labels, random_state=2021, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(train_val_masks, train_val_labels, random_state=2021, test_size=0.1)

    # Here we create TensorFlow datasets from the NumPy arrays
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_inputs, train_masks), train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices(((validation_inputs, validation_masks), validation_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices(((test_inputs, test_masks), test_labels))

    # Here we shuffle the training dataset and set the batch size
    BUFFER_SIZE = len(train_inputs) 
    BATCH_SIZE = 32
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Define the model
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    # Train the model
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=3)

    # Test the model
    test_loss, test_acc = model.evaluate(test_dataset,verbose=2)
    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    main()