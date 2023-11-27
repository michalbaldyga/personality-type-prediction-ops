# OPS Code Processing

This Python script specializes in parsing and processing Objective Personality System (OPS) codes from a CSV file. It extracts various "coins" representing different personality traits encoded within the OPS codes and saves the enhanced data in CSV format.

## Features Overview

- **Regex Patterns**: Utilizes regex patterns for validating OPS code formats and identifying incorrect cognitive functions.
- **Coin Extraction**: Implements multiple methods to extract detailed personality traits from OPS codes.
- **Data Cleaning**: Includes robust data cleaning techniques to ensure data integrity and accuracy.
- **Data Processing**: Efficiently processes OPS codes and integrates the extracted coins into the dataset.

## Detailed Method Descriptions

### Data Cleaning (`clean_ops_data`)

- **Purpose**: Cleans OPS data in the CSV file by removing extraneous annotations and standardizing OPS code formats.
- **Implementation**: 
   - Removes annotations using regex patterns.
   - Replaces placeholder characters ('?', '??') with 'x'.
   - Converts uppercase 'X' to lowercase.
   - Strips parentheses and other non-essential characters.
- **Output**: A cleaned CSV file, ready for further processing.

### Coin Extraction Methods

- `get_human_needs_coins`: Extracts three coins (Observer, Decider, Preferences) from the observing and deciding functions in the OPS code.
- `get_letter_coins`: Retrieves two coins (Observer, Decider) based on cognitive functions.
- `get_animal_coins`: Determines four coins (Energy Animal, Info Animal, Dominant Animal, Introverted vs Extraverted) from animal symbols in the OPS code.
- `get_sexual_modality_coins`: Extracts two coins (Sensory, Extraverted Decider) from the modality part of the OPS code.
- **Commonality**: Each method replaces unknown ('x', 'xx') or unspecified values with `None` to maintain data consistency.

### Data Processing (`process_ops_code_to_coins_in_csv`)

- **Purpose**: Processes OPS codes in the cleaned CSV file and enriches the dataset with the extracted coins.
- **Implementation**:
  - Parses each OPS code in the dataset.
  - Extracts coins using the defined methods.
  - Adds new columns to the dataset for each coin.
  - Handles errors and inconsistencies in OPS codes gracefully.
- **Output**: A processed CSV file with enriched data, including extracted coins for each OPS code.

## Usage

1. Ensure the CSV file with OPS codes is located in the appropriate directory (`static/csv/records.csv`).
2. Run the script to generate new CSV files with cleaned and processed data.

This script is a valuable tool for researchers and analysts working with OPS data, offering a comprehensive way to extract and analyze personality traits encoded within OPS codes.
