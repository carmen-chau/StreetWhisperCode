# StreetWhisperCode
A github directory storing the code for the STREET Whisper Transcription/Translation Application

# Description of folder structure
Currently there are 2 different Python file one can run.

Note: Both files utilizes a locally downloaded Whisper model (.pt file). Since this file is too large to upload onto Github, you would need to manually download this file. To do so, instead of doing ```whisper.load_model("copied-large-v2.pt")```, you would do ```whisper.load_model("large-v2.pt")``` to download the model onto your local system. Then, navigate to your .cache directory, and "move' this model file to your directory where the transcription/translation scripts are to use locally. 

TEMP NOTE: You can find file ```copied-large-v2.pt``` here for your reference: [INSERT GOOGLE DRIVE LINK HERE]

File #1: ```whisper_no_diarization.py``` is a file that ONLY transcribes/translates the audio file. It does NOT provide speaker diarization.

File #2: ```whisper_with_diarization.py``` is a file that, on top of what #1 does, ALSO does speaker diarization. ```whisper_with_diarization.py``` uses an additional file called ```merge_timestamps.py``` for speaker diarization. It also uses an additional file called ```config.yaml``` that contains the dependencies needed to load the diarization pipeline locally from Pyannote. 

# Instructions for running the code
Note: The instructions will most likely differ between Macbooks and Windows. Currently, we only have instructions for running on a Macbook. 

Macbook installation instructions:
- To test the functionality of version1.py, create a new virtual environment with the dependencies as stated in ```requirements.txt```
- To test the functionality of version2.py, create a new virtual environment with the dependencies as stated in ```requirements2.txt```
- Note: For both .py files, you will need to manually run ```pip commands``` in the terminal, in additional to running the usual ```pip install -r requirements.txt``` or ```pip install -r requirements2.txt```
- If the following error appears:
-   <img width="1407" alt="Screenshot 2023-11-05 at 1 25 02 AM" src="https://github.com/carmen-chau/StreetWhisperCode/assets/80921817/0d1a52f0-f01c-4075-957f-8af9b8c883f7">
- Navigate to the file that is HYPERLINKED in the error message above (it should redirect you to the mixins.py file inside your env that you are using to run the .py files)
- Make the following changes to the mixins.py file (ie: comment out Line 37, add line 38).
<img width="724" alt="Screenshot 2023-11-05 at 1 26 12 AM" src="https://github.com/carmen-chau/StreetWhisperCode/assets/80921817/7227c547-13ed-4d9d-a977-f59c4a6b4e6a">
