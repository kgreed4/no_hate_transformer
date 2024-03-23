import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the text and label from your dataset
        text = self.data.iloc[idx]['tweet']  # Adjust field name as necessary
        label = self.data.iloc[idx]['class']  # Adjust field name as necessary

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.seq_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Prepare the input dictionary
        input_ids = encoding['input_ids'].squeeze()  # Remove batch dimension added by 'return_tensors'
        attention_mask = encoding['attention_mask'].squeeze()  # Remove batch dimension added by 'return_tensors'

        return {
            'input': input_ids.long(),  # Ensure input_ids is a long tensor
            'mask': attention_mask.long(),  # Ensure attention_mask is a long tensor
            'label': torch.tensor(label).float()  # Convert label to a float tensor
        }

