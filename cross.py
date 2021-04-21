import scipy
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
import os
#import matplotlib.pyplot as plt
#import statistics as stat

path="model_pred/"
entries = os.listdir(path)
audio1 = AudioSegment.from_wav(path+entries[0])
for indice in entries[1:len(entries)]:
	print(indice)
	audio2 = AudioSegment.from_wav(path+indice)
	audio1= audio1.append(audio2, crossfade=10)
audio1.export("model_pred/file.wav",format="wav",bitrate="512k")
fs, data = wavfile.read("model_pred/file.wav")
print(data.dtype)
if data.dtype==np.int32:
	print("yes")
	data=(data>>16).astype(np.int16)
	outputch="model_pred/file.wav"
	scipy.io.wavfile.write(outputch, fs, data )

