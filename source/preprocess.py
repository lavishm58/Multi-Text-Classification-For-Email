import os 
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset, ClassLabel
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_excel('dataset/Training_data.xlsx')

df = df.dropna(axis=0)
df. drop_duplicates(subset=['Text'], inplace=True)

# Data Cleaning 
def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    return text

df['clean_text'] = df['Text'].apply(clean_text)

from nltk.tokenize import word_tokenize
nltk.download('punkt')

df['tokens'] = df['clean_text'].apply(word_tokenize)

# Removing StopWords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

lemmatizer = WordNetLemmatizer()

# Lemmatize dataset
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['clean_text'] = df['tokens'].apply(lambda x: ' '.join(x))  # Rejoin tokens to form the text

df.to_csv('Training_data_preprocessed.csv', index=False)

