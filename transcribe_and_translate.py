"""
This file contains code that attempts to transcribe AND translate audio files
"""
#Step 1: Import statements
import whisper
import time
from pyannote.audio import Pipeline
from pyannote_combo import diarize_text

#Step 2: Loads the audio file path
audio_file_path = "audio_files/eng_interview.wav"

#Step 3: Defines the name of the whisper model that is downloaded locally. Then, load this model.
#NOTE: Here, we are using Whisper large version 2
model_name = "copied-large-v2.pt"
model = whisper.load_model(model_name)

#Step 4: Define the speaker diarization pipeline using a downloaded yaml file
# NOTE: The specific pipeline used (in an offline manner) is this one: https://huggingface.co/pyannote/speaker-diarization-3.1/tree/main
speaker_diarization_pipeline = Pipeline.from_pretrained("config.yaml")

#Step 5: Transcribe the audio file path
# Note: The parameter task="translate" is OPTIONAL if the original audio file was in english
# By default, task = "translate" translates the language of the original audio file to english.
# Here, passed in task = "translate" since we are interested in transcribing a spanish interview to english
asr_result = model.transcribe(audio = audio_file_path, task = "translate", fp16=False)

#Step 6: Using diarization pipeline, run diarization
# NOTE: The pipeline should auto-detect the number of speakers in the file, but if you want to specify, can pass in addditional argument: num_speakers=2
diarization_result = speaker_diarization_pipeline("audio_files/eng_interview.wav")

#Step 7: Calling the imported pyannote_combo file's diarize_text to create a collection of strings with the timestamps, speaker identification and text.
final_result = diarize_text(asr_result, diarization_result)

# Printing purposes
for seg, spk, sent in final_result:
    line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
    print(line)

# # NOTE: STEP 8 AND 9 ARE NOT TESTED YET, SO COMMENTED OUT
# # Step 8: Create a new text file (in this sample code, the text file is called "spanish_interview.txt").
# # We write the headers "Timestamp" and "Text" with a 25 long "space gap" between the words (for formatting)
# txt_header = f"{'Timestamp:':<25} {'Speaker:':<20} Text"
# with open("combo_eng_interview.txt", "a") as txt_file:
#     txt_file.write(txt_header + "\n")
#     txt_file.close()
# #
# # # Step 9: Writing the result form Step 7 into a text file
# for seg, spk, sent in final_result:
#     with open("combo_eng_interview.txt", "a") as txt_file:
#         start_timestamp_as_time_obj = time.gmtime(seg.start)
#         converted_start_timestamp = time.strftime("%H:%M:%S",start_timestamp_as_time_obj)  # Formats the start timestamp as hour:minute:second format
#
#         end_timestamp_as_time_obj = time.gmtime(seg.end)
#         converted_end_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)  # Formats the end timestamp as hour:minute:second format
#
#         full_timestamp = converted_start_timestamp + "-" + converted_end_timestamp
#         f_string_formatted_timestamp = f"{full_timestamp:<25}"
#
#         f_string_formatted_speaker = f"{spk:<20}"
#
#         f_string_formatted_text = f"{sent}"
#
#         txt_file.write(f_string_formatted_timestamp + f_string_formatted_speaker + f_string_formatted_text + "\n")
