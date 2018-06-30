import arff
import numpy as np

row_count = 0 #sum of all frames
wav_count = 0 #sum of all files
feature_count = 13 #number of features in each row

f_input = open("_f_input.txt", "w")

prev_wav = "" #id of previous file
current_wav = "" #id of current file

row_count_2 = 0
wav_count_2 = 0

#this are the lists we will pass to tensorflow
feature_list = [] #list of lists containing the features for each frame
label_list = [] #list of lists containing the one hot vector label of each frame

def append_frame (row): #appends features and label of a frame to the corresponding lists.
	current_features = np.zeros(feature_count)
	for r in range(1,feature_count):
		current_features[r] = row[r]
	feature_list.append(current_features)
	one_hot_label = get_label(row)
	label_list.append(one_hot_label)

def get_label(row): #encode label as a one hot vector
	label = row[feature_count + 1]
	if label == 0:
		return np.array([1,0,0,0])
	elif label == 1:
		return np.array([0,1,0,0])
	elif label == 2:
		return np.array([0,0,1,0])
	elif label == 3:
		return np.array([0,0,0,1])
	else:
		return -1

def append_zero (): #append a 'zero frame'
	np.append(feature_list, [[0,0,0,0,0,0,0,0,0,0,0,0,0]], axis = 0)
	np.append(label_list, [label_list[len(label_list)-1]], axis = 0)

for current_row in arff.load('_RECOLA.arff'): #iterate over data and count number of frames and files
	row_count += 1
	current_wav = current_row[0]
	if current_wav != prev_wav:
		wav_count += 1
		prev_wav = current_wav

average_length = row_count / wav_count #avg number of rows per file

frame_count = 0
current_wav = ""
prev_wav = ""
for current_row in arff.load('_RECOLA.arff'):
	frame_count += 1
	
	current_wav = current_row[0] #get name of current file
	if prev_wav == "": #sets initial prev_wav to current_wav
		prev_wav = current_wav
	
	if current_wav == prev_wav: #check if we still consider the same file
		if frame_count <= average_length: #append frame if below 'average frame lenght'
			append_frame(current_row)
		else: #ignore frame if we already appended >= 'average frame lenght' many frames of this file
			continue

	else:	#new file
		while frame_count <= average_length: #if we added less than 'average frame lenght' many frames for the previous file: fill with zeros
			append_zero()
			frame_count += 1
		append_frame(current_row) #append first frame of the new file
		prev_wav = current_wav
		frame_count = 1
