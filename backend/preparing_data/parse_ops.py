import os
import re

import pandas as pd

# Corrected regex pattern
OPS_CODE_FORMAT = (r"([MF][MF]|[x?]{1,2})-([SNTF][ie]|[x?]{1,2})/([SNTF][ie]|[x?]{1,2})-([SPBC][SPBC]|[x?]{1,"
                   r"2})/([SPBC]|[x?]{1,2})\(([SPBC]|[x?]{1,2})\)")
# Not correct cognitive function
WRONG_COGNITIVE_FORMAT = r"(S[ei]/[S][ei])|(N[ei]/[N][ei])|(F[ei]/[F][ei])|(T[ei]/[T][ei])"


# Ten coins implemented

def replace_question_mark(value):
    return None if value in {'?', '??', 'x', 'xx'} else value


# Three coins for human needs
def get_human_needs_coins(observing_function, deciding_function):
    observing_function = replace_question_mark(observing_function)
    deciding_function = replace_question_mark(deciding_function)

    observer = 'Oi' if observing_function in ['Si', 'Ni'] else 'Oe'
    decider = 'Di' if deciding_function in ['Ti', 'Fi'] else 'De'
    preferences = 'OO' if observing_function in ['Si', 'Se', 'Ni', 'Ne'] else 'DD'

    return {
        'Observer': observer,
        'Decider': decider,
        'Preferences': preferences
    }


# Two coins for letters
def get_letter_coins(observing, deciding):
    observing = replace_question_mark(observing)
    deciding = replace_question_mark(deciding)
    # temperaments = {
    #     ('S', 'T'): 'ST',
    #     ('S', 'F'): 'SF',
    #     ('N', 'T'): 'NT',
    #     ('N', 'F'): 'NF'
    # }
    # temperament = temperaments.get((observing, deciding))
    return {
        'Observer': observing,
        'Decider': deciding,
        # 'Temperament': temperament
    }


#
# def determine_info_energy_dominance(third_animal, fourth_animal):
#     energy_animals = {'B', 'C'}
#     info_animals = {'S', 'P'}
#     animals_set = {replace_question_mark(third_animal), replace_question_mark(fourth_animal)}
#     if animals_set & energy_animals:
#         return 'Energy Dominant'
#     elif animals_set & info_animals:
#         return 'Info Dominant'
#     else:
#         return None

# Three coins for animals
def get_animal_coins(first_two_animals, third_animal, fourth_animal) -> dict:
    first_animal = replace_question_mark(first_two_animals[0])
    second_animal = replace_question_mark(first_two_animals[1])
    # third_animal = replace_question_mark(third_animal) lol it is not used ( stupid system XD)
    fourth_animal = replace_question_mark(fourth_animal)

    energy = "Sleep" if first_animal == "S" or second_animal == "S" else "Play"
    info = "Consume" if first_animal == "C" or second_animal == "C" else "Blast"

    dominant = "Info" if fourth_animal in ["S", "P"] else "Energy"
    # introversion_extroversion = "Introversion" if fourth_animal in ["P", "B"] else "Extroversion"

    return {
        'Energy Animal': energy,
        'Info Animal': info,
        'Dominant Animal': dominant,
        # 'Introversion/Extroversion': introversion_extroversion
        # 'Info vs Energy Dominant': info_vs_energy_dominant
    }


# Two coins for sexual modality
def get_sexual_modality_coins(modality) -> dict:
    sensory = replace_question_mark(modality[0])
    decider = replace_question_mark(modality[1])
    # learning_styles = {
    #     ('M', 'M'): 'Kinesthetic',
    #     ('M', 'F'): 'Audio',
    #     ('F', 'M'): 'Tester',
    #     ('F', 'F'): 'Visual',
    # }
    # learning_style = learning_styles.get((sensory, decider))
    return {
        'Sensory': sensory,
        'Extraverted Decider': decider,
        # 'Learning Style': learning_style
    }


def extract_coins_from_ops(ops: str) -> dict:
    match = re.match(OPS_CODE_FORMAT, ops)
    if not match:
        raise ValueError(f"Invalid or incomplete OPS string format: {ops}")

    wrong_cognitive = re.search(WRONG_COGNITIVE_FORMAT, ops)
    if wrong_cognitive:
        raise ValueError(f"Invalid cognitive function in OPS code : {wrong_cognitive}")

    # TODO check wrong animals

    modality, observing, deciding, first_two_animals, third_animal, fourth_animal = match.groups()

    # TODO add handling not known coins ( ??/xx/x/?)
    _parsed_ops = {
        'Human Needs': get_human_needs_coins(observing, deciding),
        'Letter': get_letter_coins(observing, deciding),
        'Animal': get_animal_coins(first_two_animals, third_animal, fourth_animal),
        'Sexual Modality': get_sexual_modality_coins(modality)
    }

    return _parsed_ops


def flatten_and_concatenate_keys(nested_dict) -> dict:
    flattened_dict = {}
    for parent_key, child_dict in nested_dict.items():
        for child_key, value in child_dict.items():
            concatenated_key = f"{parent_key}_{child_key}"
            flattened_dict[concatenated_key] = value
    return flattened_dict


def _get_path_to_csv() -> str:
    csv_rel_path = os.path.join('..', '..', 'static', 'csv', 'records_update.csv')
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, csv_rel_path)

    return csv_path if os.path.isfile(csv_path) else None


def process_ops_data():
    csv_path = _get_path_to_csv()
    if not csv_path:
        print("CSV path is not valid.")
        return
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        try:
            ops_code = row['ops']
            parsed_ops = extract_coins_from_ops(ops_code)
            coins = flatten_and_concatenate_keys(parsed_ops)

            for key, value in coins.items():
                df.at[index, key] = value
        except ValueError as e:
            print(f"Error processing row {index}: {e}")

    df.to_csv(csv_path, index=False)
