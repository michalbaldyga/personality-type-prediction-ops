import os
import pandas as pd

from backend.predict.predict import predict
from tqdm import tqdm


def _get_path_to_model() -> str:
    path = os.path.join(os.path.dirname(__file__), 'model')
    return path if os.path.isdir(path) else os.path.join(os.path.dirname(__file__), '../release', 'model')


def test_model() -> float:
    """Calculate model accuracy on the test dataset."""

    # Read test dataset
    df = pd.read_csv('../../datasets/dataset_test.csv', delimiter='|')
    df = df.dropna()  # Remove all None rows

    # Initialize accuracy variables
    correct, total = 0, len(df['text'])

    # Iterate over each example in the test dataset
    for i in tqdm(range(total)):
        # Get text and label for current example
        text = str(df['text'][i])
        label = df['label'][i]

        # Make a prediction using the classifier
        guess = predict(text)[0]['label']

        # Check if the predicted label matches the true label
        if guess == label:
            correct += 1

    # Calculate accuracy as the ratio of correct predictions to total examples
    accuracy = correct / total
    return accuracy


print(f"Accuracy: {test_model()}")
