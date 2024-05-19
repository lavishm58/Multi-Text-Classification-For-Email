import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import BertTokenizer
import torch
import json 
from config import *

# load training data
df = pd.read_csv('dataset/Training_data_preprocessed.csv')
df= df.loc[:1000,:]
# Convert categorical labels to numerical labels
category_labels = ClassLabel(names=list(df['Category'].unique()))
type_labels = ClassLabel(names=list(df['EmailType'].unique()))
df['Category'] = df['Category'].map(lambda x: category_labels.str2int(x))
df['EmailType'] = df['EmailType'].map(lambda x: type_labels.str2int(x))

# Create Dataset objects
train_dataset = Dataset.from_pandas(df)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['clean_text'], padding='max_length', truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(['Text', 'clean_text', 'tokens'])

# Set the format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Category', 'EmailType'])

