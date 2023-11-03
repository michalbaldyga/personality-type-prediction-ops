import os
import re

import pandas as pd

OPS_CODE_FORMAT = (r"([MF][MF]|[x?]{1,2})-([SNTF][ie]|[x?]{1,2})/([SNTF][ie]|[x?]{1,2})-([SPBC][SPBC]|[x?]{1,"
                   r"2})/([SPBC]|[x?]{1,2})\(([SPBC]|[x?]{1,2})\)")

WRONG_COGNITIVE_FORMAT = r"(S[ei]/[S][ei])|(N[ei]/[N][ei])|(F[ei]/[F][ei])|(T[ei]/[T][ei])"


def replace_question_mark(value):
    """
    Replace placeholders for unknown or unspecified values in the OPS code with None.

    Parameters:
    value (str): A string representing a part of the OPS code that might contain placeholders.

    Returns:
    str or None: Returns None if the input value is a placeholder, otherwise returns the input value.
    """
    return None if value in {'?', '??', 'x', 'xx'} else value


def get_human_needs_coins(observing_function, deciding_function) -> dict:
    """
    Determine the human needs coins based on the observing and deciding functions from an OPS code.

    Parameters:
    observing_function (str): The observing function part of the OPS code.
    deciding_function (str): The deciding function part of the OPS code.

    Returns:
    dict: A dictionary with keys 'Observer', 'Decider', and 'Preferences' representing human needs coins.
    """
    observing_function = replace_question_mark(observing_function)
    deciding_function = replace_question_mark(deciding_function)

    observer = None
    decider = None
    preferences = None

    if observing_function and deciding_function:
        observer = 'Oi' if observing_function in ['Si', 'Ni'] else 'Oe'
        decider = 'Di' if deciding_function in ['Ti', 'Fi'] else 'De'
        preferences = 'OO' if observing_function in ['Si', 'Se', 'Ni', 'Ne'] else 'DD'

    return {
        'Observer': observer,
        'Decider': decider,
        'Preferences': preferences
    }


def get_letter_coins(observing, deciding) -> dict:
    """
    Extract the letter-based coins from the OPS code.

    Parameters:
    observing (str): The observing trait part of the OPS code.
    deciding (str): The deciding trait part of the OPS code.

    Returns:
    dict: A dictionary with keys 'Observer' and 'Decider' representing the letter coins.
    """
    observing = replace_question_mark(observing)
    deciding = replace_question_mark(deciding)

    return {
        'Observer': observing,
        'Decider': deciding,
    }


def get_animal_coins(first_two_animals, third_animal, fourth_animal) -> dict:
    """
    Extract the animal coins from the OPS code.

    Parameters:
    first_two_animals (str): The first two animals part of the OPS code.
    third_animal (str): The third animal part of the OPS code.
    fourth_animal (str): The fourth animal part of the OPS code.

    Returns:
    dict: A dictionary with keys 'Energy Animal', 'Info Animal', and 'Dominant Animal' representing the animal coins.
    """

    first_animal = replace_question_mark(first_two_animals[0])
    second_animal = replace_question_mark(first_two_animals[1])
    fourth_animal = replace_question_mark(fourth_animal)
    energy = None
    info = None
    dominant = None
    intro_extro = None

    if first_animal and second_animal and fourth_animal:
        energy = "Sleep" if first_animal == "S" or second_animal == "S" else "Play"
        info = "Consume" if first_animal == "C" or second_animal == "C" else "Blast"

        # optional
        dominant = "Info" if fourth_animal in ["S", "P"] else "Energy"
        intro_extro = "Extro" if fourth_animal in ["C", "S"] else "Intro"

    return {
        'Energy Animal': energy,
        'Info Animal': info,
        'Dominant Animal': dominant,
        'Introverted vs Extraverted': intro_extro
    }


def get_sexual_modality_coins(modality) -> dict:
    """
    Extract the sexual modality coins from the OPS code.

    Parameters:
    modality (str): The modality part of the OPS code.

    Returns:
    dict: A dictionary with keys 'Sensory' and 'Extraverted Decider' representing the sexual modality coins.
    """
    sensory = replace_question_mark(modality[0])
    decider = replace_question_mark(modality[1])
    return {
        'Sensory': sensory,
        'Extraverted Decider': decider,
    }


def extract_coins_from_ops(ops: str) -> dict:
    """
    Parse an OPS code string and extract all the coins from it.

    Parameters:
    ops (str): The OPS code string to be parsed.

    Returns:
    dict: A nested dictionary containing all extracted coins.
    """
    match = re.match(OPS_CODE_FORMAT, ops)
    if not match:
        raise ValueError(f"Invalid or incomplete OPS string format: {ops}")

    wrong_cognitive = re.search(WRONG_COGNITIVE_FORMAT, ops)
    if wrong_cognitive:
        raise ValueError(f"Invalid cognitive function in OPS code : {wrong_cognitive}")

    modality, observing, deciding, first_two_animals, third_animal, fourth_animal = match.groups()

    _parsed_ops = {
        'Human Needs': get_human_needs_coins(observing, deciding),
        'Letter': get_letter_coins(observing, deciding),
        'Animal': get_animal_coins(first_two_animals, third_animal, fourth_animal),
        'Sexual Modality': get_sexual_modality_coins(modality)
    }

    return _parsed_ops


def flatten_and_concatenate_keys(nested_dict) -> dict:
    """
    Flatten a nested dictionary by concatenating parent and child keys.

    Parameters:
    nested_dict (dict): A nested dictionary with dictionaries as values.

    Returns:
    dict: A flattened dictionary with concatenated keys.
    """
    flattened_dict = {}
    for parent_key, child_dict in nested_dict.items():
        for child_key, value in child_dict.items():
            concatenated_key = f"{parent_key}_{child_key}"
            flattened_dict[concatenated_key] = value
    return flattened_dict


def get_path_to_csv() -> str:
    """
    Construct the file path to the CSV file containing the OPS codes.

    Returns:
    str: The file path to the CSV file, or None if the file does not exist.
    """
    csv_rel_path = os.path.join('..', '..', 'static', 'csv', 'records_update.csv')
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, csv_rel_path)

    return csv_path if os.path.isfile(csv_path) else None


def save_csv(df: pd.DataFrame, original_csv_path: str, suffix: str) -> str:
    """
    Save a DataFrame to a CSV file with a specified suffix added to the file name.

    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    original_csv_path (str): The path of the original CSV file.
    suffix (str): The suffix to add to the original file name.

    Returns:
    str: The file path to the newly saved CSV file.
    """
    new_csv_name = os.path.splitext(os.path.basename(original_csv_path))[0] + suffix + '.csv'
    new_csv_path = os.path.join(os.path.dirname(original_csv_path), new_csv_name)
    df.to_csv(new_csv_path, index=False)
    return new_csv_path


def clean_ops_data(csv_path: str) -> str:
    """
    Clean the OPS data by removing annotations and filtering for valid OPS codes.

    Parameters:
    csv_path (str): The file path to the CSV file that needs cleaning.

    Returns:
    str: The file path to the cleaned CSV file.
    """
    df = pd.read_csv(csv_path)
    annotation_pattern = r" \[\d\]$"
    df['ops'] = df['ops'].str.replace(annotation_pattern, '', regex=True)
    cleaned_df = df[df['ops'].str.match(OPS_CODE_FORMAT, na=False)]
    return save_csv(cleaned_df, csv_path, '_cleaned')


def process_ops_code_to_coins_in_csv(cleaned_csv_path: str):
    """
    Process the OPS codes in the cleaned CSV file and add extracted coins to the CSV.

    Parameters:
    cleaned_csv_path (str): The file path to the cleaned CSV file containing OPS codes.

    Returns:
    str: The file path to the processed CSV file with added coins.
    """
    cleaned_df = pd.read_csv(cleaned_csv_path)
    for index, row in cleaned_df.iterrows():
        try:
            ops_code = row['ops']
            parsed_ops = extract_coins_from_ops(ops_code)
            coins = flatten_and_concatenate_keys(parsed_ops)
            for key, value in coins.items():
                cleaned_df.at[index, key] = value if value is not None else None
        except ValueError as e:
            print(f"Error processing row {index}: {e}")
    return save_csv(cleaned_df, cleaned_csv_path, '_processed')


def main():
    """
    The main function to clean and process OPS data from a CSV file.
    """
    csv_path = get_path_to_csv()
    if not csv_path:
        print("CSV path is not valid.")
        return
    cleaned_csv_path = clean_ops_data(csv_path)
    process_ops_code_to_coins_in_csv(cleaned_csv_path)


if __name__ == "__main__":
    main()
