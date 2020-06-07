import torch
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import numpy as np
import warnings
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Sentiment(nn.Module):
    def __init__(self,bert_model):
        super(Sentiment, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=0.25)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

bert_model = BertModel.from_pretrained('bert-base-chinese')
best_model = Sentiment(bert_model).to(device)
best_model.load_state_dict(torch.load('best_model.bin'))

class GPReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_len):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }



# Create Data Loaders

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4)


def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    pred_scores = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["review"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            outputs = F.softmax(outputs, dim=1)
            # print(outputs)
            max_preds, idx_preds = torch.max(outputs, dim=1)
            # scores = outputs[:,1]
            scores = F.tanh(2*outputs[:,1]-1)
            pred_scores.extend(scores)
            review_texts.extend(texts)
            predictions.extend(idx_preds)
            prediction_probs.extend(outputs)
            break
    predictions = torch.stack(predictions).cpu()
    pred_scores = torch.stack(pred_scores).cpu()
    return review_texts, predictions, pred_scores

MAX_LEN = 256
BATCH_SIZE = 32
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
df_pred = pd.read_csv('sentence_vader_result.csv')
pred_data_loader = create_data_loader(df_pred, tokenizer, MAX_LEN, BATCH_SIZE)

y_review_texts, y_pred, scores = get_predictions(best_model,pred_data_loader)
res_df = pd.DataFrame(columns=['review','pred','score'])
res_df['review'] = y_review_texts
res_df['pred'] = y_pred
res_df['score'] = scores
print(res_df)
res_df.to_csv('bert_insurance.csv',index=False)