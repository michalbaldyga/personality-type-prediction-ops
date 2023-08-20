from datasets import ClassLabel, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import pandas as pd

MODEL_OUTPUT_DIR = "../release/model"


def load_dataset_from_csv(path):
    """Loading dataset from csv file."""
    df = pd.read_csv(path, delimiter='|')
    df = pd.DataFrame(df)
    df = df.dropna()
    return Dataset.from_pandas(df)


# Load dataset and labels
dataset = load_dataset_from_csv("../../datasets/dataset_train.csv")
dataset = dataset.train_test_split(test_size=0.2)
labels = ClassLabel(names=['OBSERVER', 'DECIDER'])

# Preprocess
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(batch):
    """Preprocessing function to tokenize text and sequences."""
    tokenized_batch = tokenizer(batch['text'], truncation=True)
    tokenized_batch['label'] = labels.str2int(batch['label'])
    return tokenized_batch


tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Evaluate
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Function to calculate the accuracy."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Train
id2label = {0: "OBSERVER", 1: "DECIDER"}
label2id = {"OBSERVER": 0, "DECIDER": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(MODEL_OUTPUT_DIR)
