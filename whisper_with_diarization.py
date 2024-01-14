"""
This file contains code that SUPPORTS SPEAKER DIARIZATION to transcribe AND translate audio files
Note: Download file dependencies from requirements2.txt
"""
#Step 1: Import statements
import whisper
import csv
import time
from pyannote.audio import Pipeline
from merge_timestamps import diarize_text
from iso639 import Lang

# Step 2: Defines the name of the whisper model that is downloaded locally. Then, load this model.
#NOTE: Here, we are using Whisper large version 2
# The model was also too large to import onto github
# You can download yourself by going to your .cache directory, making a copy of the large-v2 model, then moving it to this directory
model_name = "copied-large-v2.pt"
model = whisper.load_model(model_name)

# Step 3: Load the audio file
audio_file_path = "audio_files/B10a_trimmed_audio.mp3"
audio = whisper.load_audio(audio_file_path)
audio = whisper.pad_or_trim(audio)
# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Step 4: Print the detected language
_, probs = model.detect_language(mel)
detected_lang_code = str(max(probs, key=probs.get))
print("Detected language, " + Lang(detected_lang_code).name)

# Step 5: Transcribe
# Note: The parameter task="translate" is OPTIONAL if the original audio file was in english
# By default, task = "translate" translates the language of the original audio file to english.
# Here, passed in task = "translate" since we are interested in transcribing a spanish interview to english
#transcribe_in_autodetect_lang = False
#if transcribe_in_autodetect_lang:
asr_result_autodetect_lang = model.transcribe(audio = audio_file_path, fp16=False)
#else:
asr_result_eng = model.transcribe(audio=audio_file_path, task="translate", fp16=False)

# Step 6: Define the speaker diarization pipeline using a downloaded yaml file
# NOTE: The specific pipeline used (in an offline manner) is this one: https://huggingface.co/pyannote/speaker-diarization-3.1/tree/main
speaker_diarization_pipeline = Pipeline.from_pretrained("config.yaml")

#Step 6: Using diarization pipeline, run diarization
# NOTE: The pipeline should auto-detect the number of speakers in the file, but if you want to specify, can pass in addditional argument: num_speakers=2
diarization_result = speaker_diarization_pipeline(audio_file_path)

#Step 7: Calling the imported pyannote_combo file's diarize_text to create a collection of strings with the timestamps, speaker identification and text.
autodetect_lang_final_result = diarize_text(asr_result_autodetect_lang, diarization_result) # TODO: We can use the same object "diarization_result" since the timestamps for either ENG or autolang detection is the same
#eng_lang_final_result = diarize_text(asr_result_eng, diarization_result)


# TODO: For testing purposes
# TODO: This is the "detailed transcription code"
# csv_headers = ["Timestamps", "Speaker", "Text[Orig_lang]"]
# with open("eng_interview_custom_line_width.csv", "w") as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(csv_headers)  # Write the header row
#     for i in range(len(eng_lang_final_result)):
#         row_to_write = []
#         seg = eng_lang_final_result[i][0]
#         start_timestamp_as_time_obj = time.gmtime(float(seg.start))
#         converted_start_timestamp = time.strftime("%H:%M:%S",start_timestamp_as_time_obj)  # Formats the start timestamp as hour:minute:second format
#
#         end_timestamp_as_time_obj = time.gmtime(float(seg.end))
#         converted_end_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)  # Formats the end timestamp as hour:minute:second format
#
#         full_timestamp = converted_start_timestamp + "-" + converted_end_timestamp
#         row_to_write.append(full_timestamp)
#
#         speaker = eng_lang_final_result[i][1]
#         row_to_write.append(speaker)
#
#         original_lang_text = eng_lang_final_result[i][2]
#         row_to_write.append(original_lang_text)
#
#         #eng_text = eng_lang_final_result[i][2] #TODO: Handle irregular lengths
#         #row_to_write.append(eng_text)
#
#         csv_writer.writerow(row_to_write)

# TODO: This is the "denote timestamps through speakers" [FOR ENG LANG]
# TODO: ADD HANDLING IF NO LENGTH or when file is length of 1
csv_headers = ["Timestamps", "Speaker", "Text[Orig Lang]"]
with open("b10a_trimmed_test_club_speaker_oglang.csv", "w") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_headers)  # Write the header row

    curr_speaker = autodetect_lang_final_result[0][1]  # Denotes the speaker that is currently "speaking" in the iteration
    initial_seg = autodetect_lang_final_result[0][0]
    start_timestamp_as_time_obj = time.gmtime(float(initial_seg.start))
    beg_speaker_timestamp = time.strftime("%H:%M:%S",start_timestamp_as_time_obj)

    end_timestamp_as_time_obj = time.gmtime(float(initial_seg.end))
    end_speaker_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)

    speaker_text_seg = autodetect_lang_final_result[0][2]

    for i in range(1, len(autodetect_lang_final_result)):
        seg = autodetect_lang_final_result[i][0]
        speaker = autodetect_lang_final_result[i][1]  # Denotes the speaker that is currently "speaking" in the iteration

        if (speaker == curr_speaker) and i < len(autodetect_lang_final_result) - 1:
            speaker_text_seg = speaker_text_seg + autodetect_lang_final_result[i][2] + "\n"
            end_timestamp_as_time_obj = time.gmtime(float(seg.end))
            end_speaker_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)

        elif (speaker == curr_speaker) and i == len(autodetect_lang_final_result) - 1:
            speaker_text_seg += autodetect_lang_final_result[i][2]
            end_timestamp_as_time_obj = time.gmtime(float(seg.end))  # TODO: Used to be seg.start
            end_speaker_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)
            # In addition to the above, since we reached the end of the iteration, need to write to csv file
            row_to_write = []
            full_timestamp = beg_speaker_timestamp + "-" + end_speaker_timestamp
            row_to_write.append(full_timestamp)

            row_to_write.append(curr_speaker)

            row_to_write.append(speaker_text_seg)

            csv_writer.writerow(row_to_write)

        else:
            # if reach here, speaker changed. This is where we write to csv file

            # Step 1: define list to write values into
            row_to_write = []

            # step 2: retrieve end timestamp value and write full timestamp value to row
            full_timestamp = beg_speaker_timestamp + "-" + end_speaker_timestamp
            row_to_write.append(full_timestamp)

            # step 3: write speaker into row
            row_to_write.append(curr_speaker)

            # step 4: write the text into row
            row_to_write.append(speaker_text_seg)

            # step 5: write entire row into csv
            csv_writer.writerow(row_to_write)

            # step 6: change value of curr_speaker to speaker (officially "switching" the counter variable curr_speaker to reflect actual value from variable speaker)
            curr_speaker = speaker

            # step 7: modify the value of the beginning timestamp to reflect when the new speaker starts talking
            start_timestamp_as_time_obj = time.gmtime(float(seg.start))
            beg_speaker_timestamp = time.strftime("%H:%M:%S", start_timestamp_as_time_obj)

            # step 8: change end_speaker_timestamp value to be the most recent end timestamp (resetting value)
            end_timestamp_as_time_obj = time.gmtime(float(seg.end))
            end_speaker_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)

            # step 9: change value of speaker_text_seg to be the text seg corresponding to the speaker at this current time
            speaker_text_seg = autodetect_lang_final_result[i][2]

            if i == len(autodetect_lang_final_result) - 1:
                row_to_write = []
                full_timestamp = beg_speaker_timestamp + "-" + end_speaker_timestamp
                row_to_write.append(full_timestamp)
                row_to_write.append(speaker)
                row_to_write.append(speaker_text_seg)
                csv_writer.writerow(row_to_write)







# TODO: This is the combo code
# Step 8: Writing the result form Step 8 into a CSV file instead
# csv_headers = ["Timestamps", "Speaker", "Text[Orig_lang]", "Text[Eng]"]
# with open("spanish_interview_translated.csv", "w") as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(csv_headers)  # Write the header row
#     for i in range(len(autodetect_lang_final_result)):
#         row_to_write = []
#         seg = autodetect_lang_final_result[i][0]
#         start_timestamp_as_time_obj = time.gmtime(float(seg.start))
#         converted_start_timestamp = time.strftime("%H:%M:%S",start_timestamp_as_time_obj)  # Formats the start timestamp as hour:minute:second format
#
#         end_timestamp_as_time_obj = time.gmtime(float(seg.end))
#         converted_end_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)  # Formats the end timestamp as hour:minute:second format
#
#         full_timestamp = converted_start_timestamp + "-" + converted_end_timestamp
#         row_to_write.append(full_timestamp)
#
#         speaker = autodetect_lang_final_result[i][1]
#         row_to_write.append(speaker)
#
#         original_lang_text = autodetect_lang_final_result[i][2]
#         row_to_write.append(original_lang_text)
#
#         eng_text = eng_lang_final_result[i][2] #TODO: Handle irregular lengths
#         row_to_write.append(eng_text)
#
#         csv_writer.writerow(row_to_write)
