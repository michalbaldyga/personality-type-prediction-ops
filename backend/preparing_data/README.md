# OPS Code Processing

This Python script is designed to parse and process Objective Personality System (OPS) codes from a CSV file. The script extracts various "coins," which are categories of traits represented by the OPS codes, and then saves the enhanced data back into CSV format.

## Regex Patterns

Two regex patterns are defined to validate the OPS codes and identify any incorrect cognitive functions that may be present in the codes:

- `OPS_CODE_FORMAT`: Validates the format of the OPS code.
- `WRONG_COGNITIVE_FORMAT`: Identifies incorrect cognitive functions within the OPS codes.

## Coin Extraction Methods

The script includes several methods to extract different "coins" from the OPS codes:

- `get_human_needs_coins`: Extracts **three coins** related to human needs (Observer, Decider, Preferences) based on the observing and deciding functions in the OPS code.
- `get_letter_coins`: Retrieves **two coins** representing the observer and decider traits.
- `get_animal_coins`: Determines **three coins** (Energy Animal, Info Animal, Dominant Animal) based on the animal representations in the OPS code.
- `get_sexual_modality_coins`: Extracts **two coins** related to the sensory and extraverted decider modalities.

Each of these methods replaces unknown or unspecified values ('?', '??', 'x', 'xx') with `None`.

## Total Coins

In total, the script identifies **ten unique coins** that categorize different aspects of the OPS codes.

## Main Functions

- `clean_ops_data`: Cleans the OPS data by removing annotations and filtering for valid OPS codes.
- `process_ops_code_to_coins_in_csv`: Processes the cleaned OPS codes and adds the extracted coins to the CSV.
- `main`: The entry point of the script that orchestrates the cleaning and processing of the OPS data.

## Usage

To use this script, ensure that the CSV file with OPS codes is placed in the appropriate directory (`static/csv/records_update.csv` relative to the script's location). Run the script, and it will generate new CSV files with the **cleaned** and **processed** data.

## TODO
- check wrong animals

## Code Descriptions with Comments

Below are code segments that are currently commented out but were considered in earlier versions of the script:

```python
# Code for determining temperaments, which was considered but is not currently in use.
# temperaments = {
#     ('S', 'T'): 'ST',
#     ('S', 'F'): 'SF',
#     ('N', 'T'): 'NT',
#     ('N', 'F'): 'NF'
# }
# temperament = temperaments.get((observing, deciding))

# Code for determining dominance in information or energy, not used in the current version.
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

# Humorous comment acknowledging the redundancy of replacing a third animal, which isn't used.
# third_animal = replace_question_mark(third_animal) lol it is not used ( stupid system XD)

# Introversion/Extroversion assignment based on the fourth animal, not currently implemented.
# introversion_extroversion = "Introversion" if fourth_animal in ["P", "B"] else "Extroversion"

# Learning style determination logic based on the modality, not active in the current script.
# learning_styles = {
#     ('M', 'M'): 'Kinesthetic',
#     ('M', 'F'): 'Audio',
#     ('F', 'M'): 'Tester',
#     ('F', 'F'): 'Visual',
# }
# learning_style = learning_styles.get((sensory, decider))