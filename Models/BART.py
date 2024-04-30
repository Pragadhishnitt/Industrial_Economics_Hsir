import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BartTokenizer, BartForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_file = "/content/Data.csv"
data = pd.read_csv(csv_file)

data['sentiment'] = 'neutral'
data.loc[data['positive'] > data['negative'], 'sentiment'] = 'positive'
data.loc[data['negative'] > data['positive'], 'sentiment'] = 'negative'

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Load the pre-trained BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def tokenize_data(texts, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                      # Sentence to encode
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,   # Pad & truncate all sentences
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks
                            return_tensors = 'pt',     # Return pytorch tensors
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

train_inputs, train_masks = tokenize_data(train_df['Text'])
test_inputs, test_masks = tokenize_data(test_df['Text'])

train_labels = torch.tensor(train_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values)
test_labels = torch.tensor(test_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = BartForSequenceClassification.from_pretrained(
    "facebook/bart-large",   
    num_labels = 3, 
    output_attentions = False, 
    output_hidden_states = False, 
)

model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )

# Train the model
num_epochs = 3

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Epoch {}".format(epoch + 1)):
        optimizer.zero_grad()
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
predictions = []
true_labels = []

for batch in tqdm(test_dataloader):
    inputs = batch[0].to(device)
    masks = batch[1].to(device)
    labels = batch[2].to('cpu').numpy()
    with torch.no_grad():
        outputs = model(inputs, attention_mask=masks)
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=1).tolist())
    true_labels.extend(labels)

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")
