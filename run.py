import scipy
from scipy.io import wavfile
import numpy as np
import os
from pydub import AudioSegment
import shutil
################################################################
import os, argparse
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from model_module import *

from model import *
from model_loss import *
from model_data import *

from complex_layers.STFT import *
from complex_layers.networks import *
from complex_layers.activations import *

#######################################################################
speech_length = 16384
sampling_rate = 16000



'GET UNSEEN SPEECH FILE PATH'
def get_file_list (file_path):

      file_list = []

      for root, dirs, files in (os.walk(file_path)):
            for fname in files: 
                  if fname == "desktop.ini" or fname == ".DS_Store": continue 

                  full_fname = os.path.join(root, fname)
                  file_list.append(full_fname)

      file_list = natsort.natsorted(file_list, reverse = False)
      file_list = np.array(file_list)

      return file_list


'INFERENCE DEEP LEARNING MODEL'
def inference (path_list, save_path):

      for index1, speech_file_path in tqdm(enumerate (path_list),disable=True):
            _, unseen_noisy_speech = scipy.io.wavfile.read(speech_file_path) #7achwa ici
            restore = []
            
            for index2 in range (int(len(unseen_noisy_speech) / speech_length)):
                  split_speech = unseen_noisy_speech[speech_length * index2 : speech_length * (index2 + 1)]
                  split_speech = np.reshape(split_speech, (1, speech_length, 1))
                  enhancement_speech = model.predict([split_speech])
                  predict = np.reshape(enhancement_speech, (speech_length, 1))
                  restore.extend(predict)
            restore = np.array(restore)
            scipy.io.wavfile.write("./model_pred/" + "{:04d}".format(index1+1) + ".wav", rate = sampling_rate, data = restore)

################################################################



#import matplotlib.pyplot as plt
#import statistics as stat

file_length=16384	
################################################################
tf.random.set_seed(seed = 42)

parser = argparse.ArgumentParser(description = 'SETTING OPTION')
parser.add_argument("--model", type = str, default = "naive_dcunet16",         help = "Input model type") #naive
parser.add_argument("--load", type = str, default = "./saves/ndcunet16_1469.h5", help = "Input save model file")
parser.add_argument("--data",  type = str, default = "./datasets/fn/",    help = "Input load unseen speech")
parser.add_argument("--save",  type = str, default = "./model_pred/",      help = "Input save predict speech")
args = parser.parse_args()
	
model_type = args.model
load_file_path = args.load
test_data_path = args.data
pred_data_path = args.save

if model_type == "naive_dcunet16":
	model = Naive_DCUnet16().model()
elif model_type == "naive_dcunet20":
	model = Naive_DCUnet20().model()
elif model_type == "dcunet16":
	model = DCUnet16().model()
elif model_type == "dcunet20":
	model = DCUnet20().model()

model.load_weights(load_file_path)
###############################################################################################################################
entries=[]
with open('filename.csv') as csv_file: # change csv file name here
        for line in csv_file:
            line = line.split(",")
            entries.append(line[0].replace("\n",""))
entries_out=entries.copy()
print("TOTAL FILES : " +str(len(entries))+"\n")
for i in range(len(entries_out)):
	#entries_out[i]=entries_out[i].replace("/home/fyras/Desktop/test_filepath","./output")
	entries_out[i]=entries_out[i].replace("/home/abir/DeepSpeech/data/Acoustic","./output")
for index,namefile in tqdm(enumerate(entries)):
	
	if (os.path.exists("datasets/fn")):
		shutil.rmtree("datasets/fn")
	os.mkdir("datasets/fn")
	if (os.path.exists("model_pred/")):
		shutil.rmtree("model_pred/")
	os.mkdir("model_pred/")
	
	inputch=str(namefile)
	fs, data = wavfile.read(inputch)
	#converting to float32
	data = data/32767
	data=data.astype(np.float32)
	name=namefile.split("/")[-1].replace(".wav","")
	
	step=len(data)//file_length
	count=1
	if not(((len(data)-1)%file_length)==0):
			dist=np.zeros(file_length-(len(data)-1)%file_length)
			dist=dist.astype(data.dtype)
			data=np.append(data,dist)
	for i in range(0,step*file_length+1,file_length):
		
			tmp=data[i:i+file_length]
			outputch="datasets/fn/"+name+str(count)+".wav"
			wavfile.write(outputch, fs, tmp )
			count=count+1
	#step3
	######################################################################################################
	
	
	'READ SPEECH FILE'
	noisy_file_list = get_file_list(file_path = test_data_path)

	'INFERENCE'
	inference(path_list = noisy_file_list, save_path = pred_data_path)
	
	######################################################################################################
	
	pathc="model_pred/"
	entriesc = os.listdir(pathc)
	audio1 = AudioSegment.from_wav(pathc+entriesc[0])
	for indice in entriesc[1:len(entriesc)]:
		
		audio2 = AudioSegment.from_wav(pathc+indice)
		audio1= audio1.append(audio2, crossfade=5)#10
	
	i = entries.index(namefile)
	name = entries_out[i].replace(".wav","")
	
	audio1.export(name+".wav",format="wav",bitrate="512k")
	fs, data = wavfile.read(name+".wav")
	
	if data.dtype==np.int32: #convert from int32 to int16
		
		data=(data>>16).astype(np.int16)
		outputch=name+".wav"
		
		scipy.io.wavfile.write(outputch, fs, data )
	######################################################################################################
	
	shutil.rmtree("model_pred/")
	shutil.rmtree("datasets/fn")

