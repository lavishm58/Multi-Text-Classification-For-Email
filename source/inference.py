from model import MultiTaskBERT
from transformers import BertTokenizer
import torch
import json
import warnings
from config import *
# Hide all user warnings
warnings.simplefilter('ignore', UserWarning)

import argparse

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Inference Script with a text flag')

    # Add the text flag/argument
    parser.add_argument('--text', type=str, help='Optional text input')

    # Parse the arguments
    args = parser.parse_args()

    # Use the provided text or default message
    if not args.text:
        examples = ['this is my mail']
    else:
        examples = [args.text]

    # load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_labels_category = len(category_label_to_id)
    num_labels_type = len(type_label_to_id)
    device = 'cpu'

    # Model Loading 
    model = MultiTaskBERT.from_pretrained('bert-base-uncased', Category=num_labels_category, EmailType=num_labels_type)
    model.to(device)

    model.load_state_dict(torch.load("models/bert_preprocessed_3.bin", map_location=device))


    # Model Inference code
    results, category_results, type_results = [], [], []

    tokenized_sentences = tokenizer(examples, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors='pt')
    dic = {
              'input_ids': torch.tensor(tokenized_sentences['input_ids'], dtype=torch.long),
              'attention_mask': torch.tensor(tokenized_sentences['attention_mask'], dtype=torch.long),
              'token_type_ids': torch.tensor(tokenized_sentences['token_type_ids'], dtype=torch.long)
        } 
    outputs = model(dic['input_ids'], attention_mask=dic['attention_mask'])
    logits_category, logits_type = outputs.logits
    category_probabilities = torch.softmax(logits_category, dim=1)
    category_predicted_class = torch.argmax(category_probabilities, dim=1)

    type_probabilities = torch.softmax(logits_type, dim=1)
    type_predicted_class = torch.argmax(type_probabilities, dim=1)

    category_results.extend(category_predicted_class.cpu().numpy())
    type_results.extend(type_predicted_class.cpu().numpy())

    for cat, type in zip(category_results, type_results):
        results.append([category_label_to_id[str(cat)], type_label_to_id[str(type)] ]) 
    print(results)
    
if __name__ == '__main__':
    main()