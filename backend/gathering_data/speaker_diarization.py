import os
from collections import defaultdict

import pandas as pd
import torch
import whisper
import yt_dlp
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tqdm import tqdm

# Constants
BEST_AUDIO_FORMAT = "bestaudio/best"
PREFERRED_CODEC = "wav"
PREFERRED_QUALITY = "192"
MODEL_PRETRAINED = "pyannote/speaker-diarization-3.0"
WHISPER_VERSION = "base"
LANGUAGE = "en"
USE_AUTH_TOKEN = "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"
CSV_DIR = '../../static/csv/'
RAW_AUDIO_FILE_DIR = os.path.join("files", "audios")
CLEAN_AUDIO_FILE_DIR = os.path.join("files", "clean_audio.wav")
INPUT_CSV_FILE_DIR = os.path.join(CSV_DIR, "interview_links.csv")
OUTPUT_CSV_FILE_DIR = os.path.join("files", "transcripts.csv")


def download_audio(url):
    """Download audio from a given URL and save it to the specified output path.

    :param url: str, YouTube video URL.
    :param output_path: str, Output path to save the downloaded audio.
    """
    video_id = url.split('=')[-1]
    output_filename = os.path.join(RAW_AUDIO_FILE_DIR, f"{video_id}.wav")
    if not os.path.exists(output_filename):
        ydl_opts = {
            "format": BEST_AUDIO_FORMAT,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": PREFERRED_CODEC,
                "preferredquality": PREFERRED_QUALITY,
            }],
            "outtmpl": output_filename[:-4],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as youtube_dl:
            youtube_dl.download([url])
        print(f"Audio downloaded and saved to {output_filename}.")
    else:
        print(f"Audio already downloaded: {output_filename}")

    return output_filename


def seconds_to_milliseconds(seconds):
    """Convert seconds to milliseconds.

    :param seconds: float, time in seconds
    :return: int, equivalent time in milliseconds
    """
    return int(seconds * 1000)


def load_diarization_model(model):
    """Load a pre-trained diarization model.

    :param model: Pre-trained diarization model.
    :return: Pipeline, pre-trained diarization model.
    """
    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained(model, use_auth_token=USE_AUTH_TOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    return pipeline


def apply_diarization(video_url, audio_file, pipeline):
    """Apply diarization to the given audio file.

    :param video_url: str, URL of the video for diarization
    :param audio_file: str, path to the audio file
    :param pipeline: Pipeline, diarization model
    :return: Pipeline, diarization results
    """
    print("Applying diarization...")
    diarization = pipeline(audio_file)
    print("Diarization completed.")
    return diarization


def remove_overlapping_intervals(speaker_timestamps, max_speaker, max_intervals):
    """Remove overlapping intervals for a specified speaker.

    :param speaker_timestamps: dict, dictionary of speaker intervals
    :param max_speaker: Speaker with the maximum duration.
    :param max_intervals: list, list of intervals for the max_speaker.

    Note: This function modifies max_intervals in-place.
    """
    for speaker, intervals in speaker_timestamps.items():
        if speaker != max_speaker:
            for interval in intervals:
                for i, max_speaker_interval in enumerate(max_intervals):
                    start_1, end_1 = max_speaker_interval
                    start_2, end_2 = interval

                    if start_1 < end_2 and start_2 < end_1:
                        del max_intervals[i]

                        if start_1 < start_2:
                            max_intervals.insert(i, (start_1, start_2))
                        if end_1 > end_2:
                            max_intervals.insert(i + 1, (end_2, end_1))


def remove_temp_files(files):
    """Remove temporary files.

    :param files: list, list of file paths to be removed
    """
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def main():
    # Load models
    diarization_model = load_diarization_model(MODEL_PRETRAINED)
    whisper_model = whisper.load_model(WHISPER_VERSION, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load input .csv file
    df = pd.read_csv(INPUT_CSV_FILE_DIR, delimiter=";")

    processed_files = set()  # Keep track of already processed files
    if os.path.exists(OUTPUT_CSV_FILE_DIR):
        with open(OUTPUT_CSV_FILE_DIR, 'r') as file:
            for line in file:
                name = line.split('|')[0]
                processed_files.add(name)

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Interviews"):
        name, youtube_link = row["name"], row["interview_link"]
        if name in processed_files:
            print(f"Skipping already processed interview: {name}")
            continue

        # Apply diarization

        raw_audio_file = download_audio(youtube_link)
        diarization_results = apply_diarization(youtube_link, raw_audio_file, diarization_model)

        # Load audio file
        audio = AudioSegment.from_file(raw_audio_file)

        # Group timestamps by speakers
        speaker_timestamps = defaultdict(list)
        for turn, _, speaker in diarization_results.itertracks(yield_label=True):
            start, end = seconds_to_milliseconds(turn.start), seconds_to_milliseconds(turn.end)
            speaker_timestamps[speaker].append((start, end))

        # Find the speaker with the maximum speaking time
        max_speaker = max(speaker_timestamps, key=lambda x: sum(stop - start for start, stop in speaker_timestamps[x]))

        # Removal overlapping intervals for the selected speaker
        max_intervals = list(speaker_timestamps[max_speaker])
        remove_overlapping_intervals(speaker_timestamps, max_speaker, max_intervals)
        sorted_intervals = sorted(max_intervals, key=lambda x: x[0])

        # Segment out the parts of the audio where the selected speaker is talking
        segments = [audio[start:end] for start, end in sorted_intervals]

        # Combine segments or handle them individually
        combined = sum(segments[1:], segments[0])

        # Export the combined audio
        combined.export(CLEAN_AUDIO_FILE_DIR, format="wav")

        # Transcribe the cleaned audio
        print("Transcribing cleaned audio...")
        result = whisper_model.transcribe(audio=CLEAN_AUDIO_FILE_DIR, language=LANGUAGE)
        print("Transcription completed.")

        # Write the transcription to a file
        with open(OUTPUT_CSV_FILE_DIR, "a") as outfile:
            outfile.write(f"{name}|{result.get('text')}\n")

        # Remove unnecessary files
        remove_temp_files([CLEAN_AUDIO_FILE_DIR])


if __name__ == "__main__":
    main()
