from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from model import MultiTaskBERT
from evaluate_and_compute_metrics import evaluate_model, update_metrics
from data_load import train_dataset
from data_load import category_labels, type_labels, num_labels_category, num_labels_type
import torch.nn as nn
from tqdm import tqdm
import torch
from config import *



# Initialize KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Initialize Model metrics variables
category_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}
emailtype_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

train_category_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}
train_emailtype_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define lists to store evaluation metrics for each fold
category_accuracy_list = []
type_accuracy_list = []
category_f1_list = []
type_f1_list = []
category_precision_list = []
type_precision_list = []
EPOCHS = 1
for fold, (train_index, eval_index) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}/{NUM_FOLDS}")

    # Split the dataset into training and evaluation sets for this fold
    train_fold_dataset = train_dataset.select(train_index)
    eval_fold_dataset = train_dataset.select(eval_index)

    # Initialize model for this fold
    model = MultiTaskBERT.from_pretrained('bert-base-uncased', Category=num_labels_category, EmailType=num_labels_type)
    model.to(device)

    # Define optimizer for this fold
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop for this fold
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in tqdm(DataLoader(train_fold_dataset, batch_size=8, shuffle=True), desc=f"Training Fold {fold + 1}/{NUM_FOLDS} Epoch {epoch + 1}/{EPOCHS}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_fold_dataset)

        # Evaluation loop for this fold
        model.eval()
    metrics = evaluate_model(model, eval_fold_dataset, device, fold)
    category_metrics, emailtype_metrics = update_metrics(metrics, category_metrics, emailtype_metrics)
    train_metrics = evaluate_model(model, train_fold_dataset, device, fold)
    train_category_metrics, train_emailtype_metrics = update_metrics(train_metrics, train_category_metrics, train_emailtype_metrics)
    torch.save(model.state_dict(), f"epoch8/bert_preprocessed_{epoch + 1}.bin")
    print(f"Fold {fold + 1}, Category Train Accuracy: {train_metrics['category_accuracy']:.2f}%, Precision: {train_metrics['category_precision']:.2f}, Recall: {train_metrics['category_recall']:.2f}, F1-Score: {train_metrics['category_f1']:.2f}")
    print(f"Fold {fold + 1}, EmailType Train Accuracy: {train_metrics['emailtype_accuracy']:.2f}%, Precision: {train_metrics['emailtype_precision']:.2f}, Recall: {train_metrics['emailtype_recall']:.2f}, F1-Score: {train_metrics['emailtype_f1']:.2f}")
    print(f"Fold {fold + 1}, Category Accuracy: {metrics['category_accuracy']:.2f}%, Precision: {metrics['category_precision']:.2f}, Recall: {metrics['category_recall']:.2f}, F1-Score: {metrics['category_f1']:.2f}")
    print(f"Fold {fold + 1}, EmailType Accuracy: {metrics['emailtype_accuracy']:.2f}%, Precision: {metrics['emailtype_precision']:.2f}, Recall: {metrics['emailtype_recall']:.2f}, F1-Score: {metrics['emailtype_f1']:.2f}")
    # torch.save(model.state_dict(), f"models/bert_preprocessed_{epoch + 1}.bin")
print(f"Average Category Accuracy: {np.mean(category_metrics['accuracy']):.2f}%")
print(f"Average Category Precision: {np.mean(category_metrics['precision']):.2f}")
print(f"Average Category Recall: {np.mean(category_metrics['recall']):.2f}")
print(f"Average Category F1-Score: {np.mean(category_metrics['f1']):.2f}")