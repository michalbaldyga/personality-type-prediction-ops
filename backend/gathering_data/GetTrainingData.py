import csv

import youtube_transcript_api

FILE = '../../static/csv/records_update_cleaned_processed.csv'
OUTPUT_FILE = '../../static/csv/data_to_train.csv'

from VideoTranscriptFetcher import VideoTranscriptFetcher
from langdetect import detect
import re

vt = VideoTranscriptFetcher()


def clean_transcript(text):
    text = re.sub('\[[A-Z0-9][a-z]*\]', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('  +', ' ', text)
    return text


coins = [["OO", "DD"], ["Oi", "Oe"], ["Di", "De"],
         ["S", "N"], ["F", "T"], ["Sleep", "Play"],
         ["Consume", "Blast"], ["Fem-S", "Mas-S"],
         ["Fem-De", "Mas-De"]]


def __get_coins(elements):
    ret = []
    if elements[5] in ["Oe", "Oi"]:
        ret.append(elements[5])
    if elements[6] in ["De", "Di"]:
        ret.append(elements[6])
    if elements[7] in ["OO", "DD"]:
        ret.append(elements[7])
    if len(elements[8]) > 1 and elements[8][0] in ['T', 'N', 'S', 'F']:
        ret.append(elements[8][0])
    if len(elements[9]) > 1 and elements[9][0] in ['T', 'N', 'S', 'F']:
        ret.append(elements[9][0])
    if elements[10] in ["Sleep", "Play"]:
        ret.append(elements[10])
    if elements[11] in ["Consume", "Blast"]:
        ret.append(elements[11])
    if elements[14] in ['M', 'F']:
        ret.append('Fem-S' if elements[14] == 'F' else 'Mas-S')
    if len(elements[15]) > 0 and elements[15][0] in ['M', 'F']:
        ret.append('Fem-De' if elements[15][0] == 'F' else 'Mas-De')

    return ret


def __balance_data(data):
    results = dict()
    for c in coins:
        min_size = min(len(data[c[0]]), len(data[c[1]]))
        results[c[0]] = data[c[0]][:min_size]
        results[c[1]] = data[c[1]][:min_size]

    return results


def __save_data_to_csv(data):
    with open(OUTPUT_FILE, mode="w", newline="", encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter="|")
        csv_writer.writerow(['Coin', 'Text'])

        for coin in data:
            for text in data[coin]:
                csv_writer.writerow([coin, text])

    print(f"Data successfully saved to {OUTPUT_FILE}")


def __get_invalid_interviews():
    results = []

    with open('../../static/txt/invalid_interviews.txt', 'r') as inv_file:
        content = inv_file.readlines()
        for line in content:
            results.append(line[:-1])

    return results


def get_training_data():
    results = dict()
    for coin in coins:
        results[coin[0]] = []
        results[coin[1]] = []

    invalid_interviews = __get_invalid_interviews()

    with open(FILE) as f:
        lines = f.readlines()
        idx = 0
        errors = []

        for line in lines[1:]:
            try:
                elements = line.split(',')
                transcript = clean_transcript(vt.get_video_transcript(elements[3]))

                if len(transcript) > 0:
                    for c in __get_coins(elements):
                        results[c].append(transcript)
            except youtube_transcript_api._errors.NoTranscriptFound:
                errors.append(f"wrong lang: {elements[0]}")
                # print(f"wrong lang: {elements[0]}")
            except youtube_transcript_api._errors.TranscriptsDisabled:
                # print(f"subtitles disabled: {elements[0]}")
                errors.append(f"subtitles disabled: {elements[0]}")

            idx += 1
            if idx % 100 == 0:
                print(f"Number of extracted transcripts: {idx}")

        print(errors)

    final_results = __balance_data(results)
    __save_data_to_csv(final_results)

get_training_data()
