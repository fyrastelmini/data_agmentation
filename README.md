# data_agmentation
Requirements:
- Python >= 3.6.9 (3.7.6)  
- numpy  
- scipy  
- librosa 0.7.2  (with numba 0.48.0)  
- tensorflow >= 2.1.0  
- silence_tensorflow
- pydub

Usage: 
- copy the .csv that contains the file locations (in the first column) to directory as "filename.csv" or just change line 96 in run.py
- replicate the directories of the inputs inside of "output/"
- >python run.py
- results will be inside "output/"
