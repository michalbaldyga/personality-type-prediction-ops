import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

from backend import utils
from backend.utils import CLASS_MAPPINGS, COINS_COLUMNS, RECORDS_CLEANED_PROCESSED_CSV

# Directory for improved quality images
improved_quality_directory = os.path.join("..", "..", "..", "static", "img", "improved_quality")
model_img_accuracy_dir = os.path.join("..", "..", "..", "static", "csv", "model_img_accuracies.csv")
model_directory = os.path.join("..", "..", "release", "img")
ops_data = pd.read_csv(RECORDS_CLEANED_PROCESSED_CSV)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set the mixed precision policy
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Setup for GPU usage
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        logging.error(e)


# Function to create and compile a model
def create_model() -> Model:
    """Creates a deep learning model based on the ResNet50 architecture with modifications for binary classification.

    The model uses the ResNet50 as a base model with its top layer excluded.
    GlobalAveragePooling2D is applied to the output of the base model. Then, a dense layer with 1024 neurons and
    ReLU activation is added. The final output layer is a dense layer with a single neuron and sigmoid activation
    for binary classification. The base model layers are set as non-trainable. An RMSprop optimizer wrapped with
    mixed precision LossScaleOptimizer is used for compiling the model. The model is compiled with binary
    crossentropy loss and accuracy as the metric.

    :return: Model, the compiled Keras model ready for training.
    """
    base_model = ResNet50(weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid", dtype="float32")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    optimizer = RMSprop(learning_rate=0.001)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    """Main function for training models for personality type prediction using images.

    This function sets up the mixed precision policy and GPU configuration, loads data,
    defines class label mappings, creates a deep learning model based on ResNet50 architecture,
    configures image data augmentation, and implements KFold cross-validation.

    The training process involves balancing the dataset using SMOTE, creating a model for each
    classification category (referred to as 'coins'), and training the model using the
    preprocessed and augmented data. The best model for each category is saved for future use.
    """
    # Map class labels to numeric values
    for coin, mapping in CLASS_MAPPINGS.items():
        if coin in ops_data.columns:
            ops_data[coin] = ops_data[coin].map(mapping)

    # Image data generator configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # KFold configuration
    folds_number = 5
    fold = KFold(n_splits=folds_number, shuffle=True, random_state=42)

    # Initialize a list to store results
    results = []

    # Train models for each coin
    for coin in COINS_COLUMNS:
        logging.info(f"Training model for {coin}")

        # Initialize the best accuracy for the current coin
        best_accuracy_for_coin = 0
        best_model_for_coin = None
        fold_counter = 1
        filtered_df = ops_data[ops_data[coin].notna()]
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: utils.image_exists(row, improved_quality_directory), axis=1)]
        filtered_df["image_path"] = filtered_df["name"].apply(
            lambda x: os.path.join(improved_quality_directory, x + ".jpg"))

        for train_index, test_index in fold.split(filtered_df):
            logging.info(f"Training fold {train_index + 1}/{folds_number} for {coin}")

            train_df = filtered_df.iloc[train_index]
            valid_df = filtered_df.iloc[test_index]

            # Apply SMOTE to balance the training labels
            smote = SMOTE(random_state=42)
            y_train = train_df[coin]
            x_resampled, y_resampled = smote.fit_resample(np.arange(len(train_df)).reshape(-1, 1), y_train)

            # Create a balanced DataFrame using the indices of the resampled data
            balanced_indices = x_resampled.flatten()
            balanced_train_df = train_df.iloc[balanced_indices]

            # Convert numeric labels back to strings for compatibility with ImageDataGenerator
            balanced_train_df.loc[:, coin] = balanced_train_df[coin].astype(str)
            valid_df.loc[:, coin] = valid_df[coin].astype(str)

            # Create data generators using the balanced DataFrame
            train_generator_fold = train_datagen.flow_from_dataframe(
                dataframe=balanced_train_df,
                x_col="image_path",
                y_col=coin,
                target_size=(224, 224),
                batch_size=32,
                class_mode="binary",
            )

            valid_generator_fold = train_datagen.flow_from_dataframe(
                dataframe=valid_df,
                x_col="image_path",
                y_col=coin,
                target_size=(224, 224),
                batch_size=32,
                class_mode="binary",
            )

            # Create a new model for each coin
            model = create_model()
            # Define a temporary model checkpoint path
            temp_model_path = os.path.join(model_directory, f"temp_model_{coin}.h5")
            # Define a model checkpoint callback
            checkpoint = ModelCheckpoint(temp_model_path, monitor="val_accuracy", save_best_only=True, mode="max")

            # Train the model
            model.fit(
                train_generator_fold,
                steps_per_epoch=len(balanced_train_df) // 32,
                epochs=10,  # Adjust epochs as necessary
                validation_data=valid_generator_fold,
                validation_steps=len(valid_df) // 32,
                callbacks=[checkpoint],
            )

            # Load the best model from this fold
            best_fold_model = tf.keras.models.load_model(temp_model_path)
            loss, accuracy = best_fold_model.evaluate(valid_generator_fold, steps=len(valid_df) // 32)
            logging.info(f"Fold {fold_counter} completed for {coin} with accuracy {accuracy}")

            # If this fold's accuracy is the best so far for the coin, update the best_accuracy and save the model
            if accuracy > best_accuracy_for_coin:
                best_accuracy_for_coin = accuracy
                # Delete the previous best model for this coin
                if best_model_for_coin and fold_counter < folds_number:
                    logging.info("Removing previous model")
                    os.remove(best_model_for_coin)
                best_model_for_coin = temp_model_path
                logging.info(f"New best model found for {coin} with accuracy: {accuracy}")

            fold_counter += 1

        # After all folds are done, rename the best model file to the final model name
        final_model_path = os.path.join(model_directory, f"model_{coin}.h5")
        if best_model_for_coin and os.path.exists(best_model_for_coin):
            os.rename(best_model_for_coin, final_model_path)
            logging.info(f"Best model for {coin} saved at {final_model_path}")

        # Append the results (coin name and accuracy) to the list
        results.append({"Coin": coin, "Accuracy": best_accuracy_for_coin})

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results)

    results_df.to_csv(model_img_accuracy_dir, index=False)
    logging.info("Training completed for all coins. Results saved.")


if __name__ == "__main__":
    main()
