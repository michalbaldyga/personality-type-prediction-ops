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
from backend.utils import RECORDS_CLEANED_PROCESSED_CSV, CLASS_MAPPINGS, COINS_COLUMNS


def main():
    """Main function for training models for personality type prediction using images.

    This function sets up the mixed precision policy and GPU configuration, loads data,
    defines class label mappings, creates a deep learning model based on ResNet50 architecture,
    configures image data augmentation, and implements KFold cross-validation.

    The training process involves balancing the dataset using SMOTE, creating a model for each
    classification category (referred to as 'coins'), and training the model using the
    preprocessed and augmented data. The best model for each category is saved for future use.
    """
    # Set the mixed precision policy
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Setup for GPU usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    # Load data
    ops_data = pd.read_csv(RECORDS_CLEANED_PROCESSED_CSV)

    # Map class labels to numeric values
    for coin, mapping in CLASS_MAPPINGS.items():
        if coin in ops_data.columns:
            ops_data[coin] = ops_data[coin].map(mapping)

    # Function to create and compile a model
    def create_model() -> Model:
        """Creates a deep learning model based on the ResNet50 architecture with modifications for binary
        classification.

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
    FOLDS_NUMBER = 5
    kfold = KFold(n_splits=FOLDS_NUMBER, shuffle=True, random_state=42)

    # Directory for improved quality images
    IMPROVED_QUALITY_DIRECTORY = os.path.join("..", "..", "..", "static", "img", "improved_quality")

    # Train models for each coin
    for coin in COINS_COLUMNS:
        filtered_df = ops_data[ops_data[coin].notna()]
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: utils.image_exists(row, IMPROVED_QUALITY_DIRECTORY), axis=1)]
        filtered_df["image_path"] = filtered_df["name"].apply(
            lambda x: os.path.join(IMPROVED_QUALITY_DIRECTORY, x + ".jpg"))

        for train_index, test_index in kfold.split(filtered_df):
            train_df = filtered_df.iloc[train_index]
            valid_df = filtered_df.iloc[test_index]

            # Apply SMOTE to balance the training labels
            smote = SMOTE(random_state=42)
            y_train = train_df[coin]
            X_resampled, y_resampled = smote.fit_resample(np.arange(len(train_df)).reshape(-1, 1), y_train)

            # Create a balanced DataFrame using the indices of the resampled data
            balanced_indices = X_resampled.flatten()
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

            # Define a model checkpoint callback
            checkpoint = ModelCheckpoint(f"model_{coin}.h5", monitor="val_accuracy", save_best_only=True, mode="max")

            # Train the model
            model.fit(
                train_generator_fold,
                steps_per_epoch=len(balanced_train_df) // 32,
                epochs=10,  # Adjust epochs as necessary
                validation_data=valid_generator_fold,
                validation_steps=len(valid_df) // 32,
                callbacks=[checkpoint]
            )


if __name__ == '__main__':
    main()
