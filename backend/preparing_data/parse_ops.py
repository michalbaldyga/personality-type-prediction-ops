import os

import pandas as pd

from backend.utils import COINS_COLUMNS, RECORDS_CSV

COINS_SPLIT = 3
NUMBER_OF_LETTERS = 2
FIRST = 0
SECOND = 1


def replace_x_to_none(value: str) -> str | None:
    """Replace 'x' or 'xx' with None in OPS code values.

    :param value: str, a string representing a part of the OPS code
    :return: Optional[str], None if value is a placeholder, else the input value
    """
    return None if value in {"x", "xx"} else value


def get_human_needs_coins(observing_function: str, deciding_function: str) -> dict[str, str | None]:
    """Determine human needs coins from OPS code observing and deciding functions.

    :param observing_function: str, the observing function part of the OPS code
    :param deciding_function: str, the deciding function part of the OPS code
    :return: Dict[str, Optional[str]], dictionary with keys 'Observer', 'Decider', and 'Preferences'
    """
    observing_function = replace_x_to_none(observing_function)
    deciding_function = replace_x_to_none(deciding_function)

    observer = "Oi" if observing_function in ["Si", "Ni", "Oi"] else "Oe" if observing_function else None
    decider = "Di" if deciding_function in ["Ti", "Fi", "Di"] else "De" if deciding_function else None
    preferences = "OO" if observing_function in ["Si", "Se", "Ni", "Ne", "OO"] else "DD" if observing_function else None

    return {
        "Observer": observer,
        "Decider": decider,
        "Preferences": preferences,
    }


def get_letter_coins(cognitive_function1: str, cognitive_function2: str) -> dict[str, str | None]:
    """Extract the letter-based coins from the OPS code.

    :param cognitive_function1: str, the first cognitive function part of the OPS code
    :param cognitive_function2: str, the second cognitive function part of the OPS code
    :return: Dict[str, Optional[str]], dictionary with keys 'Observer' and 'Decider'
    """
    cognitive_function1 = replace_x_to_none(cognitive_function1)
    cognitive_function2 = replace_x_to_none(cognitive_function2)

    sensual = ["Si", "Se"]
    thinking = ["Ti", "Te"]
    human_coins = ["Oi", "Oe", "Di", "De", "OO", "DD"]

    observing = "S" if cognitive_function1 in sensual or cognitive_function2 in sensual else "N" if not (
            cognitive_function1 in human_coins or cognitive_function2 in human_coins) else None
    deciding = "T" if cognitive_function1 in thinking or cognitive_function2 in thinking else "F" if not (
            cognitive_function1 in human_coins or cognitive_function2 in human_coins) else None

    return {
        "Observer": observing,
        "Decider": deciding,
    }


def get_animal_coins(first_two_animals: str, fourth_animal: str) -> dict[str, str | None]:
    """Extract the animal coins from the OPS code.

    :param first_two_animals: str, the first two animals part of the OPS code
    :param fourth_animal: str, the fourth animal part of the OPS code
    :return: Dict[str, Optional[str]], dictionary with keys 'Energy Animal', 'Info Animal', and 'Dominant Animal'
    """
    first_animal = replace_x_to_none(first_two_animals[FIRST])
    second_animal = replace_x_to_none(first_two_animals[SECOND])
    fourth_animal = replace_x_to_none(fourth_animal)

    energy = "Sleep" if "S" in [first_animal, second_animal] else "Play" if first_animal and second_animal else None
    info = "Consume" if "C" in [first_animal, second_animal] else "Blast" if first_animal and second_animal else None
    dominant = "Info" if fourth_animal in ["S", "P"] else "Energy" if fourth_animal else None
    intro_extro = "Extro" if fourth_animal in ["C", "S"] else "Intro" if fourth_animal else None

    return {
        "Energy Animal": energy,
        "Info Animal": info,
        "Dominant Animal": dominant,
        "Introverted vs Extraverted": intro_extro,
    }


def get_sexual_modality_coins(modality: str) -> dict[str, str | None]:
    """Extract the sexual modality coins from the OPS code.

    :param modality: str, the modality part of the OPS code
    :return: Dict[str, Optional[str]], dictionary with keys 'Sensory' and 'Extraverted Decider'
    """
    sensory = replace_x_to_none(modality[FIRST])
    decider = replace_x_to_none(modality[SECOND])
    return {
        "Sensory": sensory,
        "Extraverted Decider": decider,
    }


def extract_coins_from_ops(ops: str) -> dict[str, dict[str, str | None]]:
    """Parse an OPS code string and extract all the coins from it.

    :param ops: str, the OPS code string to be parsed
    :return: Dict[str, Dict[str, Optional[str]]], a nested dictionary containing all extracted coins
    """
    parts = ops.split("-")
    if len(parts) != COINS_SPLIT:
        raise ValueError(
            f"Invalid or incomplete OPS string format: {ops} -- OPS code should have 3 parts separated by '-'.")

    modalities = 0
    modality = parts[modalities]
    if not (len(modality) == NUMBER_OF_LETTERS and all(c in "MFx" for c in modality)):
        raise ValueError(
            f"Invalid or incomplete OPS string format: {ops} -- The first part must be two characters from 'M', 'F', "
            f"'x'.")

    cognitive = 1
    second_part = parts[cognitive].split("/")
    if len(second_part) != NUMBER_OF_LETTERS:
        raise ValueError(
            f"Invalid or incomplete OPS string format: {ops} -- The second part should have 2 sections separated by '/'.")

    observing, deciding = second_part

    if not ((len(observing) == NUMBER_OF_LETTERS and observing[FIRST] in "SNTFDOx" and
             observing[SECOND] in "iex") or observing == "xx"):
        raise ValueError(
            f"Invalid or incomplete OPS string format: {ops} -- The first section of the second part is incorrect.")

    if not ((len(deciding) == NUMBER_OF_LETTERS and deciding[FIRST] in "SNTFDOx" and
             deciding[SECOND] in "iex") or deciding == "xx"):
        raise ValueError(
            f"Invalid or incomplete OPS string format: {ops} -- The second section of the second part is incorrect.")

    animals = 2
    third_part = parts[animals].split("/")
    if len(third_part) != NUMBER_OF_LETTERS:
        raise ValueError(
            f"Invalid or incomplete OPS string format: {ops} -- The third part should have 2 sections separated by '/'.")

    first_two_animals, third_and_fourth_animal = third_part

    return {
        "Human Needs": get_human_needs_coins(observing, deciding),
        "Letter": get_letter_coins(observing, deciding),
        "Animal": get_animal_coins(first_two_animals, third_and_fourth_animal[SECOND]),
        "Sexual Modality": get_sexual_modality_coins(modality),
    }


def flatten_and_concatenate_keys(nested_dict: dict[str, dict[str, str | None]]) -> dict[str, str | None]:
    """Flatten a nested dictionary by concatenating parent and child keys.

    :param nested_dict: Dict[str, Dict[str, Optional[str]]], a nested dictionary with dictionaries as values
    :return: Dict[str, Optional[str]], a flattened dictionary with concatenated keys
    """
    flattened_dict = {}
    for parent_key, child_dict in nested_dict.items():
        for child_key, value in child_dict.items():
            concatenated_key = f"{parent_key}_{child_key}"
            flattened_dict[concatenated_key] = value
    return flattened_dict


def save_csv(df: pd.DataFrame, original_csv_path: str, suffix: str) -> str:
    """Save a DataFrame to a CSV file with a specified suffix added to the file name.

    :param df: pd.DataFrame, the DataFrame to be saved
    :param original_csv_path: str, the path of the original CSV file
    :param suffix: str, the suffix to add to the original file name
    :return: str, the file path to the newly saved CSV file
    """
    old_name = 0
    new_csv_name = os.path.splitext(os.path.basename(original_csv_path))[old_name] + suffix + ".csv"
    new_csv_path = os.path.join(os.path.dirname(original_csv_path), new_csv_name)
    df.to_csv(new_csv_path, index=False)
    return new_csv_path


def clean_ops_data(csv_path: str) -> str:
    """Clean the OPS data by removing annotations, replacing '?' with 'x', converting 'X' to lowercase.

    :param csv_path: str, the file path to the CSV file that needs cleaning
    :return: str, the file path to the cleaned CSV file
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Remove annotations from the 'ops' column
    annotation_pattern = r" \[\d\]$"
    df["ops"] = df["ops"].str.replace(annotation_pattern, "", regex=True)

    # Replace '?' with 'x' in the 'ops' column
    df["ops"] = df["ops"].str.replace("?", "x", regex=False)

    # Convert all 'X' to lowercase 'x' in the 'ops' column
    df["ops"] = df["ops"].str.replace("X", "x")

    # Remove parentheses from the 'ops' column
    df["ops"] = df["ops"].str.replace("[()]", "", regex=True)

    # Define or use an existing function to save the cleaned DataFrame
    return save_csv(df, csv_path, "_cleaned")


def process_ops_code_to_coins_in_csv(cleaned_csv_path: str) -> str:
    """Process the OPS codes in the cleaned CSV file and add extracted coins.

    :param cleaned_csv_path: str, the file path to the cleaned CSV file containing OPS codes
    :return: str, the file path to the processed CSV file with coins added
    """
    cleaned_df = pd.read_csv(cleaned_csv_path)

    for key in COINS_COLUMNS:
        if key not in cleaned_df.columns:
            cleaned_df[key] = pd.Series(dtype="object")

    # List to store indices of rows to delete
    rows_to_delete = []

    for index, row in cleaned_df.iterrows():
        try:
            ops_code = row["ops"]
            parsed_ops = extract_coins_from_ops(ops_code)
            coins = flatten_and_concatenate_keys(parsed_ops)
            for key, value in coins.items():
                cleaned_df.at[index, key] = value
        except ValueError as e:
            # Mark row for deletion in case of error
            rows_to_delete.append(index)
            print(f"Error processing row {index}: {e}")

    # Drop the marked rows
    cleaned_df.drop(rows_to_delete, inplace=True)

    return save_csv(cleaned_df, cleaned_csv_path, "_processed")


def main():
    """The main function to clean and process OPS data from a CSV file."""
    csv_path = RECORDS_CSV
    if not csv_path:
        print("CSV path is not valid.")
        return
    cleaned_csv_path = clean_ops_data(csv_path)
    process_ops_code_to_coins_in_csv(cleaned_csv_path)


if __name__ == "__main__":
    main()
