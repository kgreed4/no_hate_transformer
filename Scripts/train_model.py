import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer

# Import TransformerClassifier class from model_components.py
from model_componenets import TransformerClassifier
# Import the encode_texts function from tokenizer.py
from tokinizer import encode_texts

def load_data(file_path):
    """
    Purpose: Load the data from the CSV file
    Input:
        file_path: str, path to the CSV file containing the data
    """
    return pd.read_csv(file_path)

def prepare_datasets(df, tokenizer):
    """
    Purpose: Prepare the training, validation, and test datasets
    Input:
        df: DataFrame, the DataFrame containing the data
        tokenizer: BertTokenizer object, the tokenizer to use
    """
    # Here we encode the texts using the tokenizer
    # We use the encode_texts function defined in the tokenizer.py file
    input_ids, attention_masks = encode_texts(tokenizer, df['tweet'].tolist(), max_length=128)
    # Convert the labels to a TensorFlow tensor
    labels = tf.convert_to_tensor(df['class'].values, dtype=tf.int32)

    # We must convert the input IDs, attention masks, and labels to NumPy arrays prior to splitting the data to avoid an error
    input_ids_np = input_ids.numpy()
    attention_masks_np = attention_masks.numpy()
    labels_np = labels.numpy()

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
    
    return train_dataset, validation_dataset, test_dataset

if __name__ == "__main__":
    file_path = 'cleaned_labeled_data.csv'
    df = load_data(file_path)

    # Here we load in the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Here we use the prepare_datasets function to prepare the training, validation, and test datasets
    # This function was defined in the current file
    train_dataset, validation_dataset, test_dataset = prepare_datasets(df, tokenizer)

    # Here are the parameters for the Transformer model
    num_layers = 4
    d_model = 128
    num_heads = 8
    d_ff = 512
    input_vocab_size = tokenizer.vocab_size + 2  
    maximum_position_encoding = 512
    rate = 0.1
    num_classes = df['class'].nunique()

    # We create the model by using the TransformerClassifier class defined in the model_components.py file
    model = TransformerClassifier(num_layers, d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding, rate, num_classes)

    # Here we define the optimizer, loss function, and metrics for the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Here we train the model on the training dataset and validate it on the validation dataset
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=5)

    # We can perform evaluation on the test dataset
    test_loss, test_accuracy = model.evaluate(test_dataset)

    # Print the test loss and accuracy
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
