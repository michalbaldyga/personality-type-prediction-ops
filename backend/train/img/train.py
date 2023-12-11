import logging
import os

import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

from backend import utils
from backend.utils import CLASS_MAPPINGS, COINS_COLUMNS, RECORDS_CLEANED_PROCESSED_CSV

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory paths
improved_quality_directory = os.path.join("..", "..", "..", "static", "img", "improved_quality")
model_img_accuracy_dir = os.path.join("..", "..", "..", "static", "csv", "model_img_accuracies.csv")
model_directory = os.path.join("..", "..", "release", "img")

# Load data
ops_data = pd.read_csv(RECORDS_CLEANED_PROCESSED_CSV)

# Set the mixed precision policy
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# GPU Setup
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        logging.error(e)


def preprocess_coins_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Preprocess a DataFrame by balancing the counts of two specified values in a given column.

    :param df: DataFrame, the input DataFrame.
    :param column_name: str, the name of the column to be processed.
    :return: DataFrame, the preprocessed DataFrame with balanced counts for the specified values.
    """
    value_counts = df[column_name].value_counts()
    min_count = value_counts.min()

    # Balancing the DataFrame
    return pd.concat([
        df[df[column_name] == 0].sample(min_count, random_state=42),
        df[df[column_name] == 1].sample(min_count, random_state=42),
    ])


def calculate_layers_to_freeze(num_samples, total_layers=165):
    """Calculate the number of layers to freeze based on the number of samples.

    :param num_samples: int, number of samples in the dataset.
    :param total_layers: int, total number of layers in the base model.
    :return: int, number of layers to freeze.
    """
    lower_limit = 1000
    if num_samples < lower_limit:
        return total_layers - 10
    return total_layers - 30


def create_model(train_base_model=False, num_layers_to_freeze=None):
    """Creates a deep learning model based on the ResNet50 architecture with modifications for binary classification.

    :param train_base_model: bool, if True, the base model is trained, otherwise, it's set as non-trainable.
    :param num_layers_to_freeze: int, number of layers from the top to freeze. If None, default freezing logic is applied.
    :return: Model, the compiled Keras model ready for training.
    """
    base_model = ResNet50(weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    if not train_base_model:
        if num_layers_to_freeze is not None:
            for layer in base_model.layers[:-num_layers_to_freeze]:
                layer.trainable = False
        else:
            for layer in base_model.layers[:-10]:  # Default freezing logic
                layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True

    optimizer = Adam(learning_rate=0.00005 if not train_base_model else 0.00001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Main function for training models
def main():
    """Main function for training models for personality type prediction using images.

    This function sets up the mixed precision policy and GPU configuration, loads data,
    defines class label mappings, creates a deep learning model based on ResNet50 architecture,
    configures image data augmentation, and implements KFold cross-validation.

    The training process involves balancing the dataset using SMOTE, creating a model for each
    classification category (referred to as 'coins'), and training the model using the
    preprocessed and augmented data. The best model for each category is saved for future use.
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    num_of_folds = 5
    fold = KFold(n_splits=num_of_folds, shuffle=True, random_state=42)
    results = []

    for coin in COINS_COLUMNS:
        logging.info(f"Training model for {coin}")

        if coin in ops_data.columns:
            ops_data[coin] = ops_data[coin].map(CLASS_MAPPINGS[coin])

        filtered_df = ops_data[ops_data[coin].notna()]
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: utils.image_exists(row, improved_quality_directory), axis=1)]
        filtered_df["image_path"] = filtered_df["name"].apply(
            lambda x: os.path.join(improved_quality_directory, x + ".jpg"))

        # Balance the training dataset
        filtered_df_coin = filtered_df[["image_path", coin]]

        # Balance the dataset
        balanced_df = preprocess_coins_dataframe(filtered_df_coin, coin)

        num_samples = len(balanced_df)
        num_layers_to_freeze = calculate_layers_to_freeze(num_samples)

        best_accuracy_for_coin = 0
        best_model_for_coin = None
        model_paths_to_delete = []  # List to keep track of non-best model paths

        for fold_counter, (train_index, test_index) in enumerate(fold.split(balanced_df), start=1):
            train_df = balanced_df.iloc[train_index]
            valid_df = balanced_df.iloc[test_index]

            train_df.loc[:, coin] = train_df[coin].astype(str)
            valid_df.loc[:, coin] = valid_df[coin].astype(str)

            train_generator = train_datagen.flow_from_dataframe(
                dataframe=train_df,
                x_col="image_path",
                y_col=coin,
                target_size=(224, 224),
                batch_size=16,
                class_mode="binary",
            )

            valid_generator = train_datagen.flow_from_dataframe(
                dataframe=valid_df,
                x_col="image_path",
                y_col=coin,
                target_size=(224, 224),
                batch_size=16,
                class_mode="binary",
            )

            # Create and compile the model
            model = create_model(train_base_model=False, num_layers_to_freeze=num_layers_to_freeze)

            # Define callbacks including ModelCheckpoint
            checkpoint_path = os.path.join(model_directory, f"best_model_{coin}_fold_{fold_counter}.h5")

            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max")

            early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            model.fit(
                train_generator,
                steps_per_epoch=len(train_df) // 16,
                epochs=10,
                validation_data=valid_generator,
                validation_steps=len(valid_df) // 16,
                callbacks=[checkpoint, early_stopping],
            )

            # Fine-tuning phase
            model = create_model(
                train_base_model=True,
                num_layers_to_freeze=num_layers_to_freeze,
            )
            model.load_weights(checkpoint_path)

            # Unfreeze the top layers of the model
            for layer in model.layers[:-num_layers_to_freeze]:
                layer.trainable = True

            # Recompile the model
            model.compile(optimizer=Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=["accuracy"])

            early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

            # Continue training (fine-tuning)
            model.fit(
                train_generator,
                steps_per_epoch=len(train_df) // 16,
                epochs=5,
                validation_data=valid_generator,
                validation_steps=len(valid_df) // 16,
                callbacks=[checkpoint, early_stopping],
            )

            # Evaluate the fine-tuned model
            best_fold_model = tf.keras.models.load_model(checkpoint_path)
            loss, accuracy = best_fold_model.evaluate(valid_generator, steps=len(valid_df) // 16)
            logging.info(f"Fold {fold_counter} completed for {coin} with accuracy {accuracy}")

            # Save the best model
            if accuracy > best_accuracy_for_coin:
                best_accuracy_for_coin = accuracy
                if best_model_for_coin:
                    model_paths_to_delete.append(best_model_for_coin)  # Add previous best model to delete list
                best_model_for_coin = checkpoint_path
            else:
                model_paths_to_delete.append(checkpoint_path)  # Add non-best model to delete list
        # Save the best model outside the fold loop
        final_model_path = os.path.join(model_directory, f"model_{coin}.h5")
        if best_model_for_coin and os.path.exists(best_model_for_coin):
            os.rename(best_model_for_coin, final_model_path)
            logging.info(f"Best model for {coin} saved at {final_model_path}")

        # Delete non-best fold models
        for model_path in model_paths_to_delete:
            if os.path.exists(model_path):
                os.remove(model_path)
                logging.info(f"Removed non-best model {model_path}")

        results.append({"Coin": coin, "Accuracy": best_accuracy_for_coin})

        # Save the accuracy results for all coins
    results_df = pd.DataFrame(results)
    results_df.to_csv(model_img_accuracy_dir, index=False)
    logging.info("Training completed for all coins. Results saved.")


if __name__ == "__main__":
    main()
