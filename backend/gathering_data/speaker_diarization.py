from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
from collections import defaultdict
import yt_dlp
import whisper
import os

# Constants
BEST_AUDIO_FORMAT = 'bestaudio/best'
PREFERRED_CODEC = 'wav'
PREFERRED_QUALITY = '192'
MODEL_PRETRAINED = "pyannote/speaker-diarization-3.0"
LANGUAGE = 'en'
USE_AUTH_TOKEN = ""
VIDEO_URL = "https://www.youtube.com/watch?v=4WEYbK0Do4U"
RAW_AUDIO_FILE_DIR = os.path.join("files", "raw_audio.wav")
CLEAN_AUDIO_FILE_DIR = os.path.join("files", "clean_audio.wav")
TRANSCRIPT_DIR = os.path.join("files", "transcript.txt")


def download_audio(url, output_path):
    """
    Download audio from a given URL and save it to the specified output path.

    Parameters:
    - url (str): YouTube video URL.
    - output_path (str): Output path to save the downloaded audio.
    """
    ydl_opts = {
        'format': BEST_AUDIO_FORMAT,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': PREFERRED_CODEC,
            'preferredquality': PREFERRED_QUALITY,
        }],
        'outtmpl': output_path[:-4],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as youtube_dl:
        youtube_dl.download([url])
    print(f"Audio downloaded and saved to {output_path}.")


def seconds_to_milliseconds(seconds):
    """
    Convert seconds to milliseconds.

    Parameters:
    - seconds (float): Time in seconds.

    Returns:
    - int: Equivalent time in milliseconds.
    """
    return int(seconds * 1000)


def load_and_apply_diarization(model, audio_file):
    """
    Load a pre-trained diarization model and apply it to the given audio file.

    Parameters:
    - model: Pre-trained diarization model.
    - audio_file (str): Path to the audio file.

    Returns:
    - Pipeline: Diarization results.
    """
    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained(model, use_auth_token=USE_AUTH_TOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    download_audio(VIDEO_URL, audio_file)

    print("Applying diarization...")
    diarization = pipeline(audio_file)
    print("Diarization completed.")
    return diarization


def combine_audio_segments(segments):
    """
    Combine audio segments into a single AudioSegment.

    Parameters:
    - segments (list): List of AudioSegment objects.

    Returns:
    - AudioSegment: Combined audio.
    """
    combined = sum(segments[1:], segments[0])
    return combined


def main():
    # Apply diarization
    diarization_results = load_and_apply_diarization(MODEL_PRETRAINED, RAW_AUDIO_FILE_DIR)

    # Load audio file
    audio = AudioSegment.from_file(RAW_AUDIO_FILE_DIR)

    # Group timestamps by speakers
    speaker_timestamps = defaultdict(list)
    for turn, _, speaker in diarization_results.itertracks(yield_label=True):
        start, end = seconds_to_milliseconds(turn.start), seconds_to_milliseconds(turn.end)
        speaker_timestamps[speaker].append((start, end))

    # Find the speaker with the maximum speaking time
    max_speaker = max(speaker_timestamps, key=lambda x: sum(stop - start for start, stop in speaker_timestamps[x]))

    # Removal overlapping intervals for the selected speaker
    max_intervals = list(speaker_timestamps[max_speaker])

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

    # Segment out the parts of the audio where the selected speaker is talking
    segments = [audio[start:end] for start, end in max_intervals]

    # Combine segments or handle them individually
    combined = combine_audio_segments(segments)

    # Export the combined audio
    combined.export(CLEAN_AUDIO_FILE_DIR, format="wav")

    # Transcribe the cleaned audio
    print("Loading whisper model...")
    model = whisper.load_model('base', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Transcribing cleaned audio...")
    result = model.transcribe(audio=CLEAN_AUDIO_FILE_DIR, language=LANGUAGE)
    print("Transcription completed.")

    # Write the transcription to a file
    with open(TRANSCRIPT_DIR, "w") as outfile:
        outfile.write(result.get('text'))


if __name__ == "__main__":
    main()
