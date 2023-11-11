from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("audio3.wav")

from pydub import AudioSegment

# Load your audio file
audio = AudioSegment.from_file("audio3.wav")

# List of timestamps (start and end) in milliseconds
# Example: [(start1, end1), (start2, end2), ...]
timestamps = []

def seconds_to_milliseconds(seconds):
    return int(seconds * 1000)

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    #print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    if speaker == "SPEAKER_00":
        timestamps.append((seconds_to_milliseconds(turn.start), seconds_to_milliseconds(turn.end)))

# Segment out the parts of the audio where Speaker 1 is talking
segments = [audio[start:end] for start, end in timestamps]

# Combine segments or handle them individually
combined = segments[0]
for segment in segments[1:]:
    combined += segment

# Export the combined audio
combined.export("speaker_1_audio.wav", format="wav")
