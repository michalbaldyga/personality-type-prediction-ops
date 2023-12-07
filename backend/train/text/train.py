import os

import evaluate
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from datasets import ClassLabel, Dataset

CSV_DIR = "../../../static/csv/"
CSV_WITH_COINS = os.path.join(CSV_DIR, "records_cleaned_processed.csv")
CSV_WITH_TRANSCRIPTS = os.path.join(CSV_DIR, "transcripts_cleaned.csv")
COINS = {"Human Needs_Observer": ["Oe", "Oi"],
         "Human Needs_Decider": ["De", "Di"],
         "Human Needs_Preferences": ["DD", "OO"],
         "Letter_Observer": ["S", "N"],
         "Letter_Decider": ["F", "T"],
         "Animal_Energy Animal": ["Play", "Sleep"],
         "Animal_Info Animal": ["Consume", "Blast"],
         "Animal_Dominant Animal": ["Info", "Energy"],
         "Animal_Introverted vs Extraverted": ["Extro", "Intro"],
         "Sexual Modality_Sensory": ["M", "F"],
         "Sexual Modality_Extraverted Decider": ["M", "F"]}


def preprocess_function(batch):
    """Preprocessing function to tokenize text and sequences."""
    tokenized_batch = tokenizer(batch["text"], truncation=True)
    tokenized_batch["label"] = labels.str2int(batch["label"])
    return tokenized_batch


def compute_metrics(eval_pred):
    """Function to calculate the accuracy."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_coins_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Preprocess a DataFrame by balancing the counts of two specified values in a given column.

    :param df: DataFrame, the input DataFrame.
    :param column_name: str, the name of the column to be processed.
    :return: DataFrame, the preprocessed DataFrame with balanced counts for the specified values.
    """
    df = df.dropna()

    column_counts = df[column_name].value_counts()
    difference = column_counts[COINS[column_name][0]] - column_counts[COINS[column_name][1]]

    if difference > 0:
        new_df = pd.concat([df[df[column_name] == COINS[column_name][0]].iloc[:-difference],
                            df[df[column_name] == COINS[column_name][1]]])
    elif difference < 0:
        new_df = pd.concat([df[df[column_name] == COINS[column_name][1]].iloc[:difference],
                            df[df[column_name] == COINS[column_name][0]]])
    else:
        new_df = df.copy()

    return new_df


# Load datasets
df_coins = pd.read_csv(CSV_WITH_COINS, delimiter=",")
df_transcripts = pd.read_csv(CSV_WITH_TRANSCRIPTS, delimiter="|")

final_accuracies = {}
final_hyperparams = {}

# Example of different hyperparameters to experiment with
learning_rates = [2e-5, 3e-5, 5e-5]
epochs = [3, 5, 7]
batch_sizes = [16, 32]
weight_decays = [0.01, 0.001]


for coin in COINS:
    best_model = None
    best_accuracy = -1
    best_hyperparams = {}

    for lr in learning_rates:
        for epoch in epochs:
            for batch_size in batch_sizes:
                for weight_decay in weight_decays:
                    df_coin = df_coins[["name", coin]]
                    merged_df = pd.merge(df_coin, df_transcripts, on="name", how="inner")
                    df_cleaned = preprocess_coins_dataframe(merged_df, coin)

                    data_dict = {"label": list(df_cleaned[coin]),
                                 "text": list(df_cleaned["transcript"])}

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
                        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id,
                    )
                    training_args = TrainingArguments(
                        output_dir=model_output_dir,
                        learning_rate=lr,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        num_train_epochs=epoch,
                        weight_decay=weight_decay,
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

                    # Train the model with the current set of hyperparameters
                    print(f"Training with lr: {lr}, epoch: {epoch}, batch_size: {batch_size}, weight_decay: {weight_decay}")
                    trainer.train()
                    print(f"Training done for {coin}")

                    trainer.save_model(model_output_dir)
                    print(f"{coin}_model saved")

                    # Evaluate the model
                    eval_result = trainer.evaluate()
                    current_accuracy = eval_result["eval_accuracy"]

                    # Update the best model if current is better
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_model = model
                        best_hyperparams = {
                            "learning_rate": lr,
                            "num_train_epochs": epoch,
                            "per_device_train_batch_size": batch_size,
                            "weight_decay": weight_decay
                        }

	# After all iterations
    print(f"Best model achieved an accuracy of: {best_accuracy:.4f}")
    print(f"Best Hyperparameters: {best_hyperparams}")
    final_accuracies[coin] = best_accuracy
    final_hyperparams[coin] = best_hyperparams

    # Now save the best model
    if best_model:
        best_model_output_dir = f"../release/best_model_{coin}"
        best_model.save_pretrained(best_model_output_dir)

# Print final accuracies for all models
print("\nFinal Accuracies for All Models:")
for coin, accuracy in final_accuracies.items():
    print(f"{coin}: {accuracy:.4f}")
