from data_load import category_labels, type_labels, num_labels_category, num_labels_type
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from config import *
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score

# Get performance metric for a particular epoch
def compute_metrics(logits_category, logits_type, labels_category, labels_type):
    pred_category = logits_category.argmax(dim=1).cpu()
    pred_type = logits_type.argmax(dim=1).cpu()
    labels_category = labels_category.cpu()
    labels_type = labels_type.cpu()
    category_accuracy = accuracy_score(labels_category, pred_category)
    type_accuracy = accuracy_score(labels_type, pred_type)
    category_precision = precision_score(labels_category, pred_category, average='weighted')
    type_precision = precision_score(labels_type, pred_type, average='weighted')
    category_f1 = f1_score(labels_category, pred_category, average='weighted')
    type_f1 = f1_score(labels_type, pred_type, average='weighted')
    category_recall = recall_score(labels_category, pred_category, average='weighted')
    type_recall = recall_score(labels_type, pred_type, average='weighted')

    return {
        'category_accuracy': category_accuracy,
        'emailtype_accuracy': type_accuracy,
        'category_f1': category_f1,
        'emailtype_f1': type_f1,
        'category_precision': category_precision,
        'emailtype_precision': type_precision,
        'category_recall': category_recall,
        'emailtype_recall': type_recall,

    }

# Get Inference for a batch using Dataloader
def evaluate_model(model, eval_fold_dataset, device, fold):
    total_eval_loss = 0
    all_logits_category = []
    all_logits_type = []
    all_labels_category = []
    all_labels_type = []
    for batch in tqdm(DataLoader(eval_fold_dataset, batch_size=16), desc=f"Evaluating Fold {fold + 1}/{NUM_FOLDS}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.item()
            logits_category, logits_type = outputs.logits
            all_logits_category.append(logits_category)
            all_logits_type.append(logits_type)
            all_labels_category.append(batch['Category'])
            all_labels_type.append(batch['EmailType'])

    avg_eval_loss = total_eval_loss / len(eval_fold_dataset)
    all_logits_category = torch.cat(all_logits_category, dim=0)
    all_logits_type = torch.cat(all_logits_type, dim=0)
    all_labels_category = torch.cat(all_labels_category, dim=0)
    all_labels_type = torch.cat(all_labels_type, dim=0)

    metrics = compute_metrics(all_logits_category, all_logits_type, all_labels_category, all_labels_type)
    return metrics

# Organising Model Performance Outputs
def update_metrics(metrics, category_metrics, emailtype_metrics):
    category_metrics["accuracy"].append(metrics["category_accuracy"])
    category_metrics["precision"].append(metrics["category_precision"])
    category_metrics["recall"].append(metrics["category_recall"])
    category_metrics["f1"].append(metrics["category_f1"])
    
    emailtype_metrics["accuracy"].append(metrics["emailtype_accuracy"])
    emailtype_metrics["precision"].append(metrics["emailtype_precision"])
    emailtype_metrics["recall"].append(metrics["emailtype_recall"])
    emailtype_metrics["f1"].append(metrics["emailtype_f1"])
    return category_metrics, emailtype_metrics