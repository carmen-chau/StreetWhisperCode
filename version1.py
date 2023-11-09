"""
This is the code from the Google Collab 1
"""
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import whisper

# Step 1: Choose model
model_name = "small"  # @param ["base","small", "medium", "large"]
model = whisper.load_model(model_name)

# Step 2: Define the name of the audio files that we wish to transcribe
audio_name = 'audio_files/AREA21.mp3'  # define as a string

# Step 3: Transcribe the audio
result = model.transcribe(audio_name, fp16=False)

# Step 4: Retrieve the transcription text and write it to a text file
transcription = result['text']
with open("transcription.txt", "w", encoding="utf-8") as txt:
    txt.write(transcription)
