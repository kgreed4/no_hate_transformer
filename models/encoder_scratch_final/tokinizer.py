from transformers import BertTokenizer
import tensorflow as tf
import pandas as pd

def encode_texts(tokenizer, texts, max_length=128):
    """
    Purpose: Encode the texts using the tokenizer
    Input:
        tokenizer: BertTokenizer object, the tokenizer to use
        texts: list, a list of texts to encode
        max_length: int, the maximum length of the encoded sequences 
    """
    encoding = tokenizer.batch_encode_plus(
        texts,
        # This is required to add special tokens such as the [CLS] and [SEP] tokens that indicate the start and end of a sentence
        add_special_tokens=True,
        # Here the padding variable is responsible for padding the sequences to the same length
        padding='max_length',
        # Here we define the maximum length of the sequences
        max_length=max_length,
        # Here we return the attention masks to differentiate between the actual tokens and the padded tokens
        return_attention_mask=True,
        # Here we return the tensors in TensorFlow format
        return_tensors='tf',
        # If the sequence is longer than max_length, it will be truncated
        truncation=True
    )
    # Here we return the input IDs and attention masks
    return encoding['input_ids'], encoding['attention_mask']

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv('/Users/sveerisetti/Desktop/Duke_Spring/Deep_Learning/Projects/Project_2/Data/cleaned_labeled_data.csv')
    input_ids, attention_masks = encode_texts(tokenizer, df['tweet'].tolist())
