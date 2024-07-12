import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import BertModel
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

import torch

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

converters = np.load('converter.npy',allow_pickle='TRUE').item()
item_to_idx = converters['item_to_idx']
type_to_idx = converters['type_to_idx']
idx_to_item = converters['idx_to_item']
idx_to_type = converters['idx_to_type']

def predict_keywords(question, model, tokenizer):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoding = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        item_logits, type_logits = model(input_ids, attention_mask)

    item_idx = torch.argmax(item_logits, dim=1).item()
    type_idx = torch.argmax(type_logits, dim=1).item()

    return item_idx, idx_to_item[item_idx], type_idx, idx_to_type[type_idx]

class BertForKeywordExtraction(nn.Module):
    def __init__(self, model_name, num_labels_item, num_labels_type):
        super(BertForKeywordExtraction, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.4)
        self.classifier_item = nn.Linear(self.bert.config.hidden_size, num_labels_item)
        self.classifier_type = nn.Linear(self.bert.config.hidden_size, num_labels_type)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        item_logits = self.classifier_item(pooled_output)
        type_logits = self.classifier_type(pooled_output)
        return item_logits, type_logits