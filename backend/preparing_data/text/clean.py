import os
from backend.utils import rename_to_ascii_csv

CSV_DIR = "../../../static/csv/"
INPUT_FILENAME = os.path.join(CSV_DIR, "transcripts.csv")
OUTPUT_FILENAME = os.path.join(CSV_DIR, "transcripts_cleaned.csv")

with open(INPUT_FILENAME, 'r', encoding='utf-8') as input_file:
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            is_first_occurrence = True
            modified_line = ""

            for char in line:
                if char == '|':
                    if is_first_occurrence:
                        is_first_occurrence = False
                        modified_line += char
                    else:
                        modified_line += ' '
                else:
                    modified_line += char

            output_file.write(modified_line)

rename_to_ascii_csv(OUTPUT_FILENAME, '|')
