import os

from datasets import ClassLabel, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import pandas as pd

CSV_DIR = "../../static/csv/"
CSV_WITH_COINS = os.path.join(CSV_DIR, "records_cleaned_processed.csv")
CSV_WITH_TRANSCRIPTS = os.path.join(CSV_DIR, "transcripts_cleaned.csv")
COINS = {'Human Needs_Observer': ["Oe", "Oi"],
         'Human Needs_Decider': ["De", "Di"],
         'Human Needs_Preferences': ["DD", "OO"],
         'Letter_Observer': ["S", "N"],
         'Letter_Decider': ["F", "T"],
         'Animal_Energy Animal': ["Play", "Sleep"],
         'Animal_Info Animal': ["Consume", "Blast"],
         'Animal_Dominant Animal': ["Info", "Energy"],
         'Animal_Introverted vs Extraverted': ["Extro", "Intro"],
         'Sexual Modality_Sensory': ["M", "F"],
         'Sexual Modality_Extraverted Decider': ["M", "F"]}


def preprocess_function(batch):
    """Preprocessing function to tokenize text and sequences."""
    tokenized_batch = tokenizer(batch['text'], truncation=True)
    tokenized_batch['label'] = labels.str2int(batch['label'])
    return tokenized_batch


def compute_metrics(eval_pred):
    """Function to calculate the accuracy."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_coin_data(df_coin, coin):
    df_coin = df_coin.dropna()
    coin_counts = df_coin[coin].value_counts()
    difference = coin_counts[COINS[coin][0]] - coin_counts[COINS[coin][1]]
    if difference > 0:
        df_cleaned = pd.concat([df_coin[df_coin[coin] == COINS[coin][0]].iloc[:-difference],
                                df_coin[df_coin[coin] == COINS[coin][1]]])
    elif difference < 0:
        df_cleaned = pd.concat([df_coin[df_coin[coin] == COINS[coin][1]].iloc[:difference],
                                df_coin[df_coin[coin] == COINS[coin][0]]])
    else:
        df_cleaned = df_coin.copy()
    return df_cleaned


# Load datasets
df_coins = pd.read_csv(CSV_WITH_COINS, delimiter=',')
df_transcripts = pd.read_csv(CSV_WITH_TRANSCRIPTS, delimiter='|')

for coin in COINS:
    df_coin = df_coins[["name", coin]]
    df_cleaned = preprocess_coin_data(df_coin, coin)

    merged_df = pd.merge(df_cleaned, df_transcripts, on="name", how="inner")

    data_dict = {"label": list(merged_df[coin]),
                 "text": list(merged_df["transcript"])}

    # Load current batch and labels
    dataset_batch = Dataset.from_dict(data_dict)
    dataset_batch = dataset_batch.train_test_split(test_size=0.2)

    labels = ClassLabel(names=[COINS[coin][0], COINS[coin][1]])
    id2label = {0: COINS[coin][0], 1: COINS[coin][1]}
    label2id = {COINS[coin][0]: 0, COINS[coin][1]: 1}

    # Preprocess
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_dataset = dataset_batch.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Evaluation metric
    accuracy = evaluate.load("accuracy")

    # Train
    model_output_dir = f"../release/model_{coin}"
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
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

    print(f"Start training for {coin}")
    trainer.train()
    print(f"Training done for {coin}")

    trainer.save_model(model_output_dir)
    print(f"{coin}_model saved")

    print(f"Evaluation for {coin}")
    results = trainer.evaluate()
    print(results)
