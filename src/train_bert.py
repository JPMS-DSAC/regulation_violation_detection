import pickle
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import random 


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

sentence_labels = pickle.load(open('data.pkl', 'rb'))

random.shuffle(sentence_labels)

tdata, edata = sentence_labels[:int(len(sentence_labels)*0.8)], sentence_labels[int(len(sentence_labels)*0.8):]

train_df = pd.DataFrame(tdata)
train_df.columns = ["text", "labels"]

# Preparing eval data

eval_df = pd.DataFrame(edata)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = MultiLabelClassificationArgs(num_train_epochs=10,sliding_window=True)

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel(
    "bert", "bert-base-uncased", num_labels=50,
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(
    eval_df
)


