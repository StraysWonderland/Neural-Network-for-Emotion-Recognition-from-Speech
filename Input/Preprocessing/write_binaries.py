# # Preprocessing of the training data
# Reading the training data from ARFF-file and write it into numpy binaries which contain arrays in normalized format.

import arff
import random
import numpy as np
import tensorflow as tf

FEATURE_COUNT = 13 
CORPUS_PATH = 'IEMOCAP_mfccFeatures.arff'

prev_wav = ""
current_wav = ""
count_row = 0
count_wav = 0
count_frame = 0
parsing_step = 0

labels = []
features = []

def split_dataset(dataset):
    set_train = []
    set_valid = []
    set_test = []
    size_set_train = int(len(dataset) * 0.8)
    size_set_eval = int(size_set_train * 0.1)
    for i in range(len(dataset)):
        if i < size_set_train:
            set_train.append(dataset[i])
        elif i < size_set_train + size_set_eval:
            set_valid.append(dataset[i])
        elif i >= size_set_train + size_set_eval:
            set_test.append(dataset[i])
    return set_train, set_valid, set_test

#rearrange the validation and test set to be class sensitive
def rearrange_dataset(dataset, labelset):
    set_class_0 = []
    set_class_1 = []
    set_class_2 = []
    set_class_3 = []
    label_class_0 = []
    label_class_1 = []
    label_class_2 = []
    label_class_3 = []
    for i in range(len(dataset)):
        if(np.array_equal(labelset[i], [1,0,0,0])):
            set_class_0.append(dataset[i])
            label_class_0.append(labelset[i])
        elif(np.array_equal(labelset[i], [0,1,0,0])):
            set_class_1.append(dataset[i])
            label_class_1.append(labelset[i])
        elif(np.array_equal(labelset[i], [0,0,1,0])):
            set_class_2.append(dataset[i])
            label_class_2.append(labelset[i])
        elif(np.array_equal(labelset[i], [0,0,0,1])):
            set_class_3.append(dataset[i])
            label_class_3.append(labelset[i])
    return set_class_0, set_class_1, set_class_2, set_class_3, label_class_0, label_class_1, label_class_2, label_class_3

def shuffle_dataset(set_features, set_labels):
    np_features = np.array(set_features)
    np_labels = np.array(set_labels)
    p = np.random.permutation(len(set_features))
    return np_features[p], np_labels[p]

def append_frame (row):
    current_features = np.empty(FEATURE_COUNT)
    for r in range(1,FEATURE_COUNT):
        current_features[r] = row[r]
    # generates a normalized array but formatted as a list consisting of a single list ([normalized_current_features])
    # so we need to access the normalized current_features at index 0 to get the current_features array.
    features.append(tf.keras.utils.normalize(current_features)[0])
    one_hot_label = get_label(row)
    labels.append(one_hot_label)

#encode label as a one hot vector
def get_label(row):
    label = row[FEATURE_COUNT + 1]
    if label == "0":
        return [1,0,0,0]
    elif label == "1":
        return [0,1,0,0]
    elif label == "2":
        return [0,0,1,0]
    elif label == "3":
        return [0,0,0,1]
    else:
        return [0,0,0,0]

def append_zero (): #append a 'zero frame'
    features.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    labels.append([labels[len(labels)-1]])


# -- Begin preprocessing --

for current_row in arff.load(CORPUS_PATH): #iterate over data and count number of frames and files
    count_row += 1
    current_wav = current_row[0]
    if current_wav != prev_wav:
        count_wav += 1
        prev_wav = current_wav

#avg number of frames per file sample
avg_frames = int(count_row / count_wav)
print("avg_frames: " + str(avg_frames) + ", the other ones: " + str(count_row) + ", " + str(count_wav))
##print "Frames per file: " + str(avg_frames)

#reset the values in current_wav and prev_wav before looping over the rows again
current_wav = ""
prev_wav = ""

for current_row in arff.load(CORPUS_PATH):
    count_frame += 1
    parsing_step += 1
    #Simply to observe the progress
    if parsing_step % 1000 == 0:
        print(str(int((parsing_step / count_row)*100)) + "%")

    current_wav = current_row[0] #Get name of current file
    if prev_wav == "":
        prev_wav = current_wav

    if current_wav == prev_wav:
        if count_frame <= avg_frames:
            append_frame(current_row)
        else:
            continue
    elif current_wav != prev_wav:
        #Fill with zeroes until there are avg_frames many frames 
        while count_frame <= avg_frames:
            append_zero()
            count_frame += 1
        #Finally, append first frame of the new file
        append_frame(current_row)
        prev_wav = current_wav
        count_frame = 1

    if parsing_step == count_row: #last file with less frames than average
        while count_frame < avg_frames: #if we added less than 'average frame lenght' many frames for the previous file: fill with zeros
            append_zero()
            count_frame += 1

#reduce labels to number of samples
#TODO: DOUBLE CHECK ON THIS
labels_reduced = []
labels_reduced.append(labels[0])
for i in range(1, count_wav):
    labels_reduced.append(labels[i*avg_frames])

#generate suitable feature array for shuffling
features_clustered = np.array(features)
features_clustered = np.reshape(features_clustered, (-1, avg_frames, len(features[0])))

#shuffle arrays
features_clustered, labels_reduced = shuffle_dataset(features_clustered, labels_reduced)

#split into training, validation and test set
features_train, features_valid, features_test = split_dataset(features_clustered)
labels_train, labels_valid, labels_test = split_dataset(labels_reduced)

#rearrange validation and test to each of their classes
validation_class_0, validation_class_1, validation_class_2, validation_class_3, val_label_class_0, val_label_class_1, val_label_class_2, val_label_class_3 = rearrange_dataset(features_valid, labels_valid)
test_class_0, test_class_1, test_class_2, test_class_3, test_label_class_0, test_label_class_1, test_label_class_2, test_label_class_3 = rearrange_dataset(features_test, labels_test)

#write features back in proper format
features_train = np.reshape(features_train, (-1, len(features[0])))
features_valid = np.reshape(features_valid, (-1, len(features[0])))
features_test = np.reshape(features_test, (-1, len(features[0])))
validation_class_0 = np.reshape(validation_class_0, (-1, len(features[0])))
validation_class_1 = np.reshape(validation_class_1, (-1, len(features[0])))
validation_class_2 = np.reshape(validation_class_2, (-1, len(features[0])))
validation_class_3 = np.reshape(validation_class_3, (-1, len(features[0])))
test_class_0 = np.reshape(test_class_0, (-1, len(features[0])))
test_class_1 = np.reshape(test_class_1, (-1, len(features[0])))
test_class_2 = np.reshape(test_class_2, (-1, len(features[0])))
test_class_3 = np.reshape(test_class_3, (-1, len(features[0])))

#write into binaries
print("Frames per sample: " + str(avg_frames))
np.save(CORPUS_PATH[:5]+'_feature_train', features_train)
np.save(CORPUS_PATH[:5]+'_feature_valid', features_valid)
np.save(CORPUS_PATH[:5]+'_feature_test', features_test)
np.save(CORPUS_PATH[:5]+'_labels_train', labels_train)
np.save(CORPUS_PATH[:5]+'_labels_valid', labels_valid)
np.save(CORPUS_PATH[:5]+'_labels_test', labels_test)
np.save(CORPUS_PATH[:5]+'_feature_valid_class_0', validation_class_0)
np.save(CORPUS_PATH[:5]+'_feature_valid_class_1', validation_class_1)
np.save(CORPUS_PATH[:5]+'_feature_valid_class_2', validation_class_2)
np.save(CORPUS_PATH[:5]+'_feature_valid_class_3', validation_class_3)
np.save(CORPUS_PATH[:5]+'_label_valid_class_0', val_label_class_0)
np.save(CORPUS_PATH[:5]+'_label_valid_class_1', val_label_class_1)
np.save(CORPUS_PATH[:5]+'_label_valid_class_2', val_label_class_2)
np.save(CORPUS_PATH[:5]+'_label_valid_class_3', val_label_class_3)
np.save(CORPUS_PATH[:5]+'_feature_test_class_0', test_class_0)
np.save(CORPUS_PATH[:5]+'_feature_test_class_1', test_class_1)
np.save(CORPUS_PATH[:5]+'_feature_test_class_2', test_class_2)
np.save(CORPUS_PATH[:5]+'_feature_test_class_3', test_class_3)
np.save(CORPUS_PATH[:5]+'_label_test_class_0', test_label_class_0)
np.save(CORPUS_PATH[:5]+'_label_test_class_1', test_label_class_1)
np.save(CORPUS_PATH[:5]+'_label_test_class_2', test_label_class_2)
np.save(CORPUS_PATH[:5]+'_label_test_class_3', test_label_class_3)
print("Preprocessing complete")
