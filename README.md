# StreetWhisperCode
A github directory storing the code for the STREET Whisper Transcription/Translation Application

# Description of folder structure
Currently there are 2 different Python file one can run.

__Note__: Both files utilizes a locally downloaded Whisper model (.pt file). Since this file is too large to upload onto Github, you would need to manually download this file. 
To download these .pt files you would do ```whisper.load_model("large-v2.pt")``` to download the model onto your local system. Then, navigate to your .cache directory, and "move' this model file to your directory where the transcription/translation scripts are to use locally. 

__Tip for MacOS__: For MacOS, the command to navigate to the downloaded .pt files (from your base directory is ```cd .cache/whisper```

__Tip for Windows / Linux__: TBD

File #1: ```whisper_no_diarization.py``` is a file that ONLY transcribes/translates the audio file. It does NOT provide speaker diarization.

File #2 [DEPRECIATED]: ```whisper_with_diarization.py``` is a file that, on top of what #1 does, ALSO does speaker diarization. ```whisper_with_diarization.py``` uses an additional file called ```merge_timestamps.py``` for speaker diarization. It also uses an additional file called ```config.yaml``` that contains the dependencies needed to load the diarization pipeline locally from Pyannote. 

File #3 [IN PROGRESS]: ```whisper_with_diarization_as_methods.py```is a file that has the same capabilities as File #2. However, it now utilizes WhisperX speaker diarization features to speed up runtime. In addition, the code is now blocked into methods for reusability and readability. 

# Instructions for running the code
Note: The instructions will most likely differ between Macbooks and Windows. Currently, we only have instructions for running on a Macbook. 

***Macbook installation instructions:***

It is recommended to download all dependencies onto a Python virtual environment (venv). Your environment CANNOT be a conda one, since the whisper dependencies are not supported there. 
Please configure your venv to support Python 3.9. Other versions of Python have not been tested yet. 

To run ```whisper_no_diarization.py```, download the dependencies found in ```requirements.txt```

To run ```whisper_with_diarization.py```, download the dependencies found in ```requirements2.txt```

[IN PROGRESS] To run ```whisper_with_diarization_as_methods.py```, download the dependencies found in ```requirements2.txt```

***Windows/Linux installation instructions:***
TBD
