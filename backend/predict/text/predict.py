import os

from transformers import pipeline

MODEL_DIR = "../../release/"
COINS = ["Human Needs_Observer", "Human Needs_Decider", "Human Needs_Preferences",
         "Letter_Observer", "Letter_Decider",
         "Animal_Energy Animal", "Animal_Info Animal", "Animal_Dominant Animal", "Animal_Introverted vs Extraverted",
         "Sexual Modality_Sensory", "Sexual Modality_Extraverted Decider"]


def _get_path_to_model(coin: str) -> str:
    model_name = f"best_model_{coin}"
    return os.path.join(MODEL_DIR, model_name)


def predict(text: str) -> dict:
    """Generate sentiment predictions for a given text across multiple ops coins.

    This function utilizes a text classification model for sentiment analysis on text.
    It predicts the sentiment for each specified coin and returns the results in a dictionary.

    :param text: str, the input text for sentiment analysis.
    :return: dict, a dictionary containing sentiment predictions for each specified coin.
    """
    predictions_dict = {}
    for coin in COINS:
        try:
            classifier = pipeline(
                "text-classification",
                model=_get_path_to_model(coin),
                tokenizer="distilbert-base-uncased",
                framework="pt",
                top_k=2,
            )
        except OSError:
            classifier = None

        if classifier is not None:
            # split text to batches of 512 tokens
            batch_size = 512
            batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]

            # predict
            predictions = {}
            for batch in batches:
                result = classifier(batch, padding=True)
                for res in result[0]:
                    predictions[res["label"]] = predictions.get(res["label"], 0) + res["score"]

            best_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
            score = 0.
            for prediction in best_predictions:
                score += prediction[1]

            predictions_dict[coin] = []
            for prediction in best_predictions:
                predictions_dict[coin].append({"label": str(prediction[0]),
                                               "percent": f"{round((prediction[1] / score) * 100, 1)}%"})
    return predictions_dict
