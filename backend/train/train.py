from datasets import ClassLabel, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import pandas as pd

COINS_NUMBER = 9
VALUES_PER_COIN = 2
COINS = [["OO", "DD"], ["Oi", "Oe"], ["Di", "De"],
         ["S", "N"], ["F", "T"], ["Sleep", "Play"],
         ["Consume", "Blast"], ["Fem-S", "Mas-S"],
         ["Fem-De", "Mas-De"]]


def load_dataset_from_csv(csv_path):
    """Loading dataset from csv file."""
    df = pd.read_csv(csv_path, delimiter='|')
    df = pd.DataFrame(df)
    return Dataset.from_pandas(df)


def preprocess_function(batch):
    """Preprocessing function to tokenize text and sequences."""
    tokenized_batch = tokenizer(batch['text'], truncation=True)
    tokenized_batch['labels'] = labels.str2int(batch['labels'])
    return tokenized_batch


def compute_metrics(eval_pred):
    """Function to calculate the accuracy."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def calculate_coins_indexes(coin_col):
    coins_indexes = []
    changes_cnt = 0
    curr_coin = coin_col[0]
    for idx, coin in enumerate(coin_col):
        if coin != curr_coin:
            curr_coin = coin
            changes_cnt += 1
        if changes_cnt == VALUES_PER_COIN:
            coins_indexes.append(idx)
            changes_cnt = 0
    coins_indexes.append(idx + 1)
    return coins_indexes


# Load dataset
dataset = load_dataset_from_csv("../../datasets/data_to_train.csv")
coins_indexes = [0] + calculate_coins_indexes(dataset['labels'])

for (coin, idx) in zip(COINS, range(0, COINS_NUMBER)):

    # Load current batch and labels
    dataset_batch = Dataset.from_dict(dataset[coins_indexes[idx]:coins_indexes[idx+1]])
    dataset_batch = dataset_batch.train_test_split(test_size=0.2)

    labels = ClassLabel(names=[coin[0], coin[1]])
    id2label = {0: coin[0], 1: coin[1]}
    label2id = {coin[0]: 0, coin[1]: 1}

    # Preprocess
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_dataset = dataset_batch.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Evaluation metric
    accuracy = evaluate.load("accuracy")

    # Train
    model_output_dir = f"../release/model_{idx}"
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=model_output_dir,
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
    trainer.save_model(model_output_dir)
