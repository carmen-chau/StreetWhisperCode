"""
File description: This file contains the code used to run transcription on audio files

Notes:
    1. Code purposefully NOT organized into methods, will create another file that organizes code into methods (for preparation to merge with the Electron code).
"""

import whisper
import time

# Step 1: Choose model (by default, choose the large_v2 model)
# Need to configure variable model_name to be customizable based on user desired input.
model_name = "large-v2" #Possible parameters:['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']
whisper_model = whisper.load_model(model_name)

#Step 2: Input audio. ASSUME for now that audio will be .wav type
# Note: A sample file name is passed in
audio_file_path = "audio_files/spanish_interview.wav"

#Step 3: Transcribe
# Note: The parameter task="translate" is OPTIONAL if the original audio file was in english
# By default, task = "translate" translates the language of the original audio file to english.
transcription = whisper_model.transcribe(audio = audio_file_path, task = "translate", fp16=False)

# Step 4: Get the "segments" from the transcription
list_of_all_segments = transcription["segments"]

# Step 5: Create a new text file (in this sample code, the text file is called "spanish_interview.txt").
# We write the headers "Timestamp" and "Text" with a 25 long "space gap" between the words (for formatting)
txt_header = f"{'Timestamp:':<25} Text"
with open("spanish_interview.txt", "a") as txt_file:
    txt_file.write(txt_header + "\n")
    txt_file.close()

#Step 6: Iterate through the dictionary of segments (and open the text file from text 5 for writing purposes)
for i in range(len(list_of_all_segments)):
    with open("spanish_interview.txt", "a") as txt_file:
        start_timestamp = list_of_all_segments[i]["start"]
        start_timestamp_as_time_obj = time.gmtime(start_timestamp)
        converted_start_timestamp = time.strftime("%H:%M:%S",start_timestamp_as_time_obj) #Formats the start timestamp as hour:minute:second format

        end_timestamp = list_of_all_segments[i]["end"]
        end_timestamp_as_time_obj = time.gmtime(end_timestamp)
        converted_end_timestamp = time.strftime("%H:%M:%S",end_timestamp_as_time_obj) #Formats the end timestamp as hour:minute:second format

        full_timestamp = converted_start_timestamp + "-" + converted_end_timestamp

        f_string_formatted_timestamp = f"{full_timestamp:<25}"
        timestamp_text = list_of_all_segments[i]["text"]
        f_string_formatted_timestamp_text = f"{timestamp_text}"
        txt_file.write(f_string_formatted_timestamp + f_string_formatted_timestamp_text + "\n") # Writes both the timestamps + the transcribed text as 1 line to the text file

        #txt_file.write((str(list_of_all_segments[i]["start"]) + "- " + str(list_of_all_segments[i]["end"]) + ": " + list_of_all_segments[i]["text"] + "\n"))
        #print("Segment text: ", list_of_all_segments[i]["text"])
        #print("Segment start time: ", list_of_all_segments[i]["start"])
        #print("Segment end time: ", list_of_all_segments[i]["end"])
    txt_file.close()
#print(transcription_dict["segments"])
