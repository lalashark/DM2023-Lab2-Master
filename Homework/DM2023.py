#!/usr/bin/env python
# coding: utf-8

# ### Student Information
# Name:詹敬群
# 
# Student ID:112291524
# 
# GitHub ID:lalashark
# 
# Kaggle name:V111621
# 
# Kaggle private scoreboard snapshot:
# 
# [Snapshot](img/pic0.png)

# ---

# ### Instructions

# 1. First: __This part is worth 30% of your grade.__ Do the **take home** exercises in the DM2023-Lab2-master. You may need to copy some cells from the Lab notebook to this notebook. 
# 
# 
# 2. Second: __This part is worth 30% of your grade.__ Participate in the in-class [Kaggle Competition](https://www.kaggle.com/t/09b1d0f3f8584d06848252277cb535f2) regarding Emotion Recognition on Twitter by this link https://www.kaggle.com/t/09b1d0f3f8584d06848252277cb535f2. The scoring will be given according to your place in the Private Leaderboard ranking: 
#     - **Bottom 40%**: Get 20% of the 30% available for this section.
# 
#     - **Top 41% - 100%**: Get (60-x)/6 + 20 points, where x is your ranking in the leaderboard (ie. If you rank 3rd your score will be (60-3)/6 + 20 = 29.5% out of 30%)   
#     Submit your last submission __BEFORE the deadline (Dec. 27th 11:59 pm, Wednesday)_. Make sure to take a screenshot of your position at the end of the competition and store it as '''pic0.png''' under the **img** folder of this repository and rerun the cell **Student Information**.
#     
# 
# 3. Third: __This part is worth 30% of your grade.__ A report of your work developping the model for the competition (You can use code and comment it). This report should include what your preprocessing steps, the feature engineering steps and an explanation of your model. You can also mention different things you tried and insights you gained. 
# 
# 
# 4. Fourth: __This part is worth 10% of your grade.__ It's hard for us to follow if your code is messy :'(, so please **tidy up your notebook** and **add minimal comments where needed**.
# 
# 
# Upload your files to your repository then submit the link to it on the corresponding e-learn assignment.
# 
# Make sure to commit and save your changes to your repository __BEFORE the deadline (Dec. 31th 11:59 pm, Sunday)__. 

# ## lets run trough the data frist

# ### see what does the tweet looks 

# In[ ]:


import json
import pandas as pd




# In[ ]:


file = open("../data/tweets_DM.json", 'r', encoding='utf-8')
tweets_id = []
tweets_hashtags = []
tweets_text = []
for line in file.readlines():
    dic = json.loads(line)
    tweets_id.append(dic["_source"]["tweet"]["tweet_id"])
    tweets_hashtags.append(dic["_source"]["tweet"]["hashtags"])
    tweets_text.append(dic["_source"]["tweet"]["text"])

src = pd.DataFrame([], columns=[]) 

src = src.assign(id = tweets_id, hashtags = tweets_hashtags, text = tweets_text)

src.head(10)


# In[ ]:


src.iloc[0]['text']


# ### the emotion part

# In[ ]:


colnames=['id', 'emotion'] 

label = pd.read_csv('../data/emotion.csv', names=colnames, header=0)


label.head(10)


# ### merge it

# In[ ]:


train_df = pd.merge(src, label, on="id", how="left")

train_df.head(10)


# In[ ]:


#we use the merge so the missing part might be NaN in 'emotion'
train_df = train_df.dropna(how='any')
train_df.head(10)


# In[ ]:


train_df.groupby(['emotion']).count()['text']


# In[ ]:


col=['id', 'emotion']
test = pd.read_csv('../data/sampleSubmission.csv', names=col, header=0)
test.head(10)


# In[ ]:


test_df = pd.merge(src, test, on="id", how="left")
test_df = test_df.dropna(axis=0, how='any') 

test_df.head(10)


# In[ ]:


test_df = test_df.drop(['hashtags', 'emotion'], axis=1)

test_df


# In[ ]:


print(len(test_df))


# In[ ]:


print(len(train_df))


# In[ ]:




import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm

import horovod.torch as hvd
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

# In[ ]:


df = train_df

df = df.drop(['hashtags'], axis=1)

df.columns = ['id', 'text', 'category']

possible_labels = df.category.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

df['label'] = df.category.replace(label_dict)


# In[ ]:





# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.15, 
                                                  random_state=17, 
                                                  stratify=df.label.values)


# In[ ]:


df['data_type'] = ['not_set']*df.shape[0]


# In[ ]:


df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'


# In[ ]:


df.groupby(['category', 'label', 'data_type']).count()


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)


# In[ ]:


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)



# In[ ]:


print(len(dataset_train), len(dataset_val))


# In[ ]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                            num_labels=len(label_dict),
                            output_attentions=False,
                            output_hidden_states=False)
model.cuda()

# In[ ]:


batch_size = 32

# dataloader_train = DataLoader(dataset_train, 
#                 sampler=RandomSampler(dataset_train), 
#                 batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                  sampler=SequentialSampler(dataset_val), 
                  batch_size=batch_size)

train_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset_train, num_replicas=hvd.size(), rank=hvd.rank())
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)


# In[ ]:


optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)
     


# In[ ]:


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
     


# In[ ]:


seed_val = 7
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(device)


# In[ ]:


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[ ]:


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    cnt = 0
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.cuda() for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        # print("model forward")
        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        # print("model backward")
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # print("model optimize")
        optimizer.step()
        # print("scheduler step")
        scheduler.step()
        
        if hvd.local_rank() == 0:
            print(f"Epoch {epoch}:  batch #{cnt}  loss: {loss.item()/len(batch):.3f}")
        cnt += 1
        # progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

         
        
    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    print(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)             
    tqdm.write(f'Training loss: {loss_train_avg}')
    print(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    print(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
    print(f'F1 Score (Weighted): {val_f1}')


# In[ ]:


test = test_df.set_index('id').T.to_dict('list')


# In[ ]:


label = []
for id in test:
  sentence = test[id]

  inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=256, return_tensors="pt")

  # to gpu
  ids = inputs["input_ids"].to(device)
  mask = inputs["attention_mask"].to(device)

  # to model
  outputs = model(ids, mask)
  logits = outputs[0]

  active_logits = logits.view(-1, model.num_labels) 
  flattened_predictions = torch.argmax(active_logits, axis=1) 

  tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
  ids_to_labels = {'0':'anticipation', '1':'sadness', '2':'fear', '3':'joy', '4':'anger', '5':'trust', '6':'disgust', '7':'surprise'}
  token_predictions = ids_to_labels[str(flattened_predictions.cpu().numpy()[0])]
  label.append(token_predictions)


# In[ ]:


fin_df = test_df

fin_df = fin_df.assign(emotion = label)

fin_df = fin_df.drop(['text'], axis=1)


# In[ ]:


fin_df.to_csv('submission.csv', index=False)


# In[ ]:




