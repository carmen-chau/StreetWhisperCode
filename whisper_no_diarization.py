"""
File description: This file contains the code used to run transcription/translation on audio files
It DOES NOT SUPPORT SPEAKER DIARIZATION.
Note: Download file dependencies from requirements.txt
"""
from typing import Any, Optional

import whisper
import time

# OVERALL TODO: Figure out how to incoperate the code for the UI buttons "Upload File" and "Output Folder"
# TODO: What does the UI mean by "live transcription"?


def define_whisper_model(model_path: str):
    """
    This method downloads a Whisper model by loading in a .pt file in the directory
    specified by parameter model_path

    This essentilly "choses" the model used for transcription/translation

    For reference, here are possible Whisper model sizes:
    ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small',
    'medium.en', 'medium', 'large-v1', 'large-v2', 'large']

    NOTE: The file being locaed MUST be a .pt file

    Preconditions:
        - model_path is a valid path to a .pt file

    :param model_path: Local path of the Whisper model
    :return: A Whisper Model Object
    """
    # TODO: Have zero idea what happens when an invalid model path is passed, need to look into that
    # TODO #2: Also, what is the return type of the model?
    whisper_model = whisper.load_model(model_path)
    return whisper_model

def transcribe_audio(whisper_model: Any, audio_file_path: str, is_translate: Optional[bool] = False):
    """
    This method takes an audio file with the path as specified by parameter audio_file_path.
    It also takes a boolean is_translate. If true, we wish to translate the transcribed text to English.
    If this is false, then we transcribe the audio file based on the autodetected language

    It then passes in both of these variables to the whisper model's transcribe method.

    Finally, a list of segments containing the timestamps and transcribed/translated text is extracted

    :param audio_file_path: str
    :param is_translate: bool
    :return: Any

    Preconditions:
        - Audio file path is defined and links to a .wav file.
    """
    if is_translate:
        transcription = whisper_model.transcribe(audio = audio_file_path, task = "translate", fp16=False)
    else:
        transcription = whisper_model.transcribe(audio = audio_file_path, fp16=False)

    return transcription["segments"]

def writing_to_file(txt_file_path: str, segment_list) -> None:
    """
    This method takes a parameter denoting the path of the txt file (parameter txt_file_path),
    and a parameter denoting a segment list object (parameter segment_list)

    It creates a txt file that includes the start and end timestamps with the corresponding transcribed/translatedd text
    :param txt_file_path:
    :param segment_list:
    :return: None
    """
    txt_header = f"{'Timestamp:':<25} Text"
    with open(txt_file_path, "a") as txt_file:
        txt_file.write(txt_header + "\n")
        txt_file.close()

    for i in range(len(segment_list)):
        with open(txt_file_path, "a") as txt_file:
            start_timestamp = segment_list[i]["start"]
            start_timestamp_as_time_obj = time.gmtime(start_timestamp)
            converted_start_timestamp = time.strftime("%H:%M:%S",start_timestamp_as_time_obj) #Formats the start timestamp as hour:minute:second format

            end_timestamp = segment_list[i]["end"]
            end_timestamp_as_time_obj = time.gmtime(end_timestamp)
            converted_end_timestamp = time.strftime("%H:%M:%S",end_timestamp_as_time_obj) #Formats the end timestamp as hour:minute:second format

            full_timestamp = converted_start_timestamp + "-" + converted_end_timestamp

            f_string_formatted_timestamp = f"{full_timestamp:<25}"
            timestamp_text = segment_list[i]["text"]
            f_string_formatted_timestamp_text = f"{timestamp_text}"
            txt_file.write(f_string_formatted_timestamp + f_string_formatted_timestamp_text + "\n") # Writes both the timestamps + the transcribed text as 1 line to the text file

        txt_file.close()

if __name__ == "__main__":
    loaded_whisper_model = define_whisper_model("copied-large-v2.pt")
    retrieved_list_audio_seg = transcribe_audio(loaded_whisper_model, "audio_files/eng_interview.wav")
    writing_to_file("eng_interview.txt", retrieved_list_audio_seg)
