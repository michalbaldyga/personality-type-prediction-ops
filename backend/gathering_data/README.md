# Gathering Data Scripts

This directory contains scripts for gathering data from SubjectivePersonalities and YouTube. These scripts get persons with their OPS types, images, interviews.

## Overview

Scripts get data from SubjectivePersonalities (person name, OPS type, image), search for interviews and process them using diarization and transcription.

## Prerequisites

- python 3.10
- torch
- pandas
- whisper (OpenAI)
- tqdm
- YoutubeDL
- PyAnnote
- pydub
- selenium

Ensure these libraries are installed before running scripts

## Scripts Overview

1. **SubjectivePersonality**: Gather needed data from SubjectivePersonality page: person name, OPS type, image, OPS community membership and sometimes link to interview.

2. **get_interviews**: Search interview for people that has no interview assigned on SubjectivePersonality page. It uses filters like name validation and checking if there is interview related keyword in video title using YoutubeBrowser class.

3. **speaker_diarization**: Process all interviews from dataset to get transcript of analyzed person statements:
- download audio file from YouTube video
- using diarization, extracts only the statements of one person
- transcript obtained statements

4. **get_interviews_time**: Calculate how long all interviews take in total to estimate how long will take diarization and transcription of all interviews in dataset


## Usage

To execute the scripts, navigate to directory containing these scripts and run python command, for example for get_interviews script:

```bash
python get_interviews.py