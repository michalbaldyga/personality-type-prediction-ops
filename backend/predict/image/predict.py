import os

import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.mixed_precision import set_global_policy
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from backend import utils
from backend.utils import RECORDS_CLEANED_PROCESSED_CSV

STATIC_IMG_DIR = os.path.join("..", "..", "..", "static", "img")
IMPROVED_QUALITY_DIRECTORY = os.path.join(STATIC_IMG_DIR, "improved_quality")

# Set the mixed precision policy
set_global_policy("mixed_float16")

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

ops_data = pd.read_csv(RECORDS_CLEANED_PROCESSED_CSV)


# Function to create and compile a model
def create_model():
    """Create and compile a ResNet50 model for binary classification.

    The function initializes a ResNet50 model, adds custom top layers, and compiles it
    with RMSprop optimizer and binary crossentropy loss function.

    Returns:
        A compiled Keras Model instance.
    """
    base_model = ResNet50(weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid", dtype="float32")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    # Use the RMSprop optimizer with a loss scale to handle mixed precision training
    optimizer = RMSprop(learning_rate=0.001)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Compile the model with the mixed precision optimizer
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Coins columns
coins_columns = [
    "Human Needs_Observer", "Human Needs_Decider", "Human Needs_Preferences",
    "Letter_Observer", "Letter_Decider", "Animal_Energy Animal",
    "Animal_Info Animal", "Animal_Dominant Animal",
    "Animal_Introverted vs Extraverted", "Sexual Modality_Sensory",
    "Sexual Modality_Extraverted Decider",
]

# Create and compile separate models for each coin
models = {coin: create_model() for coin in coins_columns}

# Set up ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Iterate over each coin and train a model
for coin in coins_columns:
    # Filter out NaN values and check if image exists with .jpg extension
    filtered_df = ops_data[ops_data[coin].notna()]
    filtered_df = filtered_df[
        filtered_df.apply(lambda row: utils.image_exists(row, IMPROVED_QUALITY_DIRECTORY), axis=1)]

    # Generate file paths with .jpg extension for the ImageDataGenerator
    filtered_df["image_path"] = filtered_df["name"].apply(
        lambda x: os.path.join(IMPROVED_QUALITY_DIRECTORY, x + ".jpg"))

    # Create a generator that reads images from the paths, and uses the coin labels for training
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=filtered_df,
        x_col="image_path",  # Updated to use the path with the .jpg extension
        y_col=coin,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        validate_filenames=True,  # Now we can set to True because we are sure that files exist
    )

    # Define a model checkpoint callback to save the best model for each coin
    checkpoint = ModelCheckpoint(f"model_{coin}.h5", monitor="accuracy", save_best_only=True, mode="max")

    # Train the model
    models[coin].fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=10,  # Adjust the number of epochs as necessary
        callbacks=[checkpoint],
    )
