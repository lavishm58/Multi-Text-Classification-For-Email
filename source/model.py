from transformers import BertTokenizer, BertModel, BertPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

# Custom Model Class
class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config, Category, EmailType):
        super().__init__(config)
        self.Category = Category
        self.EmailType = EmailType
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_category = nn.Linear(config.hidden_size, Category)
        self.classifier_type = nn.Linear(config.hidden_size, EmailType)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, Category=None, EmailType=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits_category = self.classifier_category(pooled_output)
        logits_type = self.classifier_type(pooled_output)

        loss = None
        if Category is not None and EmailType is not None:
            # add class weights here

            loss_fct_category = nn.CrossEntropyLoss()
            loss_fct_type = nn.CrossEntropyLoss()
            
            # loss_fct = nn.CrossEntropyLoss()
            loss_category = loss_fct_category(logits_category, Category)
            loss_type = loss_fct_type(logits_type, EmailType)
            loss = loss_category + loss_type

        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits_category, logits_type),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
