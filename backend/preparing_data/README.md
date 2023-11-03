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
- `get_animal_coins`: Determines **four coins** (Energy Animal, Info Animal, Dominant Animal, Introverted vs Extraverted) based on the animal representations in the OPS code.
- `get_sexual_modality_coins`: Extracts **two coins** related to the sensory and extraverted decider modalities.
Each of these methods replaces unknown or unspecified values ('?', '??', 'x', 'xx') with `None`.

## Total Coins
In total, the script identifies **eleven unique coins** that categorize different aspects of the OPS codes.

## Main Functions
- `clean_ops_data`: Cleans the OPS data by removing annotations and filtering for valid OPS codes.
- `process_ops_code_to_coins_in_csv`: Processes the cleaned OPS codes and adds the extracted coins to the CSV.
- `main`: The entry point of the script that orchestrates the cleaning and processing of the OPS data.

## Usage
To use this script, ensure that the CSV file with OPS codes is placed in the appropriate directory (`static/csv/records_update.csv` relative to the script's location). Run the script, and it will generate new CSV files with the **cleaned** and **processed** data.


