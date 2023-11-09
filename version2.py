"""
This is the **modified code* based on the Google Collab 2
"""
# Import statements
import whisper
import datetime
import subprocess
import torch
# NEW IMPORT BELOW
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Step 1: Choose model
model_name = "tiny"  # @param ["base","small", "medium", "large"]
model = whisper.load_model(model_name)

# Step 2: Define pre-trained speaker recognition model
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cpu"))
# NOTE: In the original Google Collab file, argument to method was: device=torch.device("cuda")
# Carmen: I use a Macbook which does not have CUDA. Thus. I had to change it to "cpu" instead
# On Windows computer, I am not sure whether calling device=torch.device("cuda") would work.

# Step 3: Define the name of the audio files that we wish to transcribe
audio_name = "audio_files/pokimane_valk.mp3"  # Try passing in "audio_files/pokimane_valk.mp3" and see if transcription still works
index_of_latest_backslash_char = audio_name.rfind("/")
index_of_latest_dot_char = audio_name.rfind(".")
input_audio_name = audio_name[index_of_latest_backslash_char + 1: index_of_latest_dot_char]

# Step 4: Define the number of speakers, language and size of the language model used for transcription
num_speakers = 2  # @param {type:"integer"}
language = 'English'  # @param ['any', 'English', 'Spanish']
model_size = 'base'  # @param ['tiny', 'base', 'small', 'medium', 'large']
model_name = model_size
if language == 'English' and model_size != 'large':
    model_name += '.en'

    # Step 5: Actual speaker analysis

    # If the audio file isn't a .wav file, convert it to the .wav format
    if audio_name[-3:] != 'wav':
        destination_folder = "audio_files"  # Temporary placeholder folder location name. In actual app, this may be an user input field
        name_wav_audio_file = destination_folder + "/" + input_audio_name + ".wav"  # Obtain the new audio file name
        subprocess.call(['ffmpeg', '-i', audio_name, name_wav_audio_file, '-y'])  # Converting audio file to .wav

    else:
        name_wav_audio_file = audio_name
        destination_folder = "audio_files"  # Temporary placeholder folder location name. In actual app, this may be an user input field

    mono_name_wav_audio_file = destination_folder + "/" + "mono_" + input_audio_name + "testing_version" + ".wav"

    subprocess.call(['ffmpeg', '-n', '-i', name_wav_audio_file, '-ac', '1', mono_name_wav_audio_file])
    # In above line, if the audio has mono audio, convert audio to stereo in order for transcription to occur properly

    path = mono_name_wav_audio_file
    # Returns a dictionary with three key-value pairs with the following keys: "text", "segments", "language"
    result = model.transcribe(path)
    # Retrieves the segmented transcription and time code data (segments is a dictionary)
    segments = result["segments"]

    # Step 6: Opens the .wav audio file and obtains the duration of the audio
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Step 7: Define segment_embedding function
    audio = Audio()
    def segment_embedding(segment):
        """
        Returns an embedding computed by the model that represents the
        <segment> as a vector.
        """
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(path, clip)
        return embedding_model(waveform[None])
        # used to be: return embedding_model(waveform[None])

    # Step 8: Pass in audio segment info in <segments> into segment_embedding
    # to be turned into embeddings and then stored into a matrix
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)
        # print(segment_embedding(segment))
    embeddings = np.nan_to_num(embeddings)

    # Step 9: Create <num_speakers> clusters from the data in embeddings
    # and iterate through each audio segment info in <segments> and
    # create new key-value pair (key: "speaker", value: "SPEAKER <label number>")
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    # Step 10: Create transcription file
    def time(secs):
        return datetime.timedelta(seconds=round(secs))

    text_file_name = "transcript_poki_rae_2_ppl"
    full_text_file_path = text_file_name + ".txt"
    f = open(full_text_file_path, "w")

    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')
    f.close()
