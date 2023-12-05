import logging
import os
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
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
    df = df.dropna()

    value_counts = df[column_name].value_counts()
    min_count = value_counts.min()

    # Balancing the DataFrame
    balanced_df = pd.concat([
        df[df[column_name] == 0].sample(min_count, random_state=42),
        df[df[column_name] == 1].sample(min_count, random_state=42)
    ])

    return balanced_df


# Function to create and compile a model
def create_model(train_base_model=False):
    base_model = ResNet50(weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    if not train_base_model:
        for layer in base_model.layers:
            layer.trainable = False

    optimizer = Adam(learning_rate=0.0001 if not train_base_model else 0.00001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Main function for training models
def main():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    fold = KFold(n_splits=5, shuffle=True, random_state=42)
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

        best_accuracy_for_coin = 0
        best_model_for_coin = None

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
                batch_size=32,
                class_mode="binary",
            )

            valid_generator = train_datagen.flow_from_dataframe(
                dataframe=valid_df,
                x_col="image_path",
                y_col=coin,
                target_size=(224, 224),
                batch_size=32,
                class_mode="binary",
            )

            model = create_model(train_base_model=False)
            temp_model_path = os.path.join(model_directory, f"temp_model_{coin}.h5")
            checkpoint = ModelCheckpoint(temp_model_path, monitor="val_accuracy", save_best_only=True, mode="max")

            model.fit(
                train_generator,
                steps_per_epoch=len(train_df) // 32,
                epochs=10,
                validation_data=valid_generator,
                validation_steps=len(valid_df) // 32,
                callbacks=[checkpoint],
            )

            # Fine-tuning phase
            model = create_model(train_base_model=True)
            for layer in model.layers[:165]:
                layer.trainable = False
            for layer in model.layers[165:]:
                layer.trainable = True
            model.compile(optimizer=Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=["accuracy"])

            model.fit(
                train_generator,
                steps_per_epoch=len(train_df) // 32,
                epochs=5,  # Fewer epochs for fine-tuning
                validation_data=valid_generator,
                validation_steps=len(valid_df) // 32,
                callbacks=[checkpoint],
            )

            best_fold_model = tf.keras.models.load_model(temp_model_path)
            loss, accuracy = best_fold_model.evaluate(valid_generator, steps=len(valid_df) // 32)
            logging.info(f"Fold {fold_counter} completed for {coin} with accuracy {accuracy}")

            if accuracy > best_accuracy_for_coin:
                best_accuracy_for_coin = accuracy
                if best_model_for_coin and fold_counter < 5:
                    os.remove(best_model_for_coin)
                best_model_for_coin = temp_model_path
                logging.info(f"New best model found for {coin} with accuracy: {accuracy}")

        final_model_path = os.path.join(model_directory, f"model_{coin}.h5")
        if best_model_for_coin and os.path.exists(best_model_for_coin):
            os.rename(best_model_for_coin, final_model_path)
            logging.info(f"Best model for {coin} saved at {final_model_path}")

        results.append({"Coin": coin, "Accuracy": best_accuracy_for_coin})

    results_df = pd.DataFrame(results)
    results_df.to_csv(model_img_accuracy_dir, index=False)
    logging.info("Training completed for all coins. Results saved.")


if __name__ == "__main__":
    main()
