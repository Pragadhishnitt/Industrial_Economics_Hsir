import pandas as pd
from sklearn.model_selection import train_test_split
from fastai.text.all import *

csv_file = "/content/Data.csv"
data = pd.read_csv(csv_file)

data['sentiment'] = 'neutral'
data.loc[data['positive'] > data['negative'], 'sentiment'] = 'positive'
data.loc[data['negative'] > data['positive'], 'sentiment'] = 'negative'

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
dls_lm = TextDataLoaders.from_df(train_df, is_lm=True, valid_pct=0.1, text_col='Text')

# Fine-tune the pre-trained language model
learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult=0.3, metrics=[accuracy, Perplexity()]).to_fp16()
learn.fit_one_cycle(1, 2e-2)

learn.save_encoder('finetuned_encoder')
dls_clas = TextDataLoaders.from_df(train_df, text_col='Text', label_col='sentiment', valid_pct=0.2, text_vocab=dls_lm.vocab)
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()
learn = learn.load_encoder('finetuned_encoder')

# Train the classifier
learn.fit_one_cycle(1, 2e-2)

# Evaluate the classifier
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

print("Validation Accuracy:", interp.print_classification_report())
