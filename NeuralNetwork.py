
# coding: utf-8

# # Neural Network for Emotion Recognition from Speech
# Recognizes emotion by classifying binary  Valence/arousal values for given utterances.
# Provide input data by extracting 13MFCC features (via OpenSMILE) and export to binary files.
# Network consists of one convolution layer
# 
# Created in cooperation with:
# [nymvno](https://github.com/nymvno) 
# &
# [StrohmFn](https://github.com/StrohmFn) 

import numpy as np
import tensorflow as tf
import random
import math

# load input data
filepath = 'Input'

# load IEMOCAP input and labels to numpy arrays
samples_train_ENG = np.load(filepath + '/IEMOCAP/IEMOCAP_feature_train.npy')
labels_train_ENG = np.load(filepath + '/IEMOCAP/IEMOCAP_labels_train.npy')

# load IMEOCAP testset
samples_test_ENG =  np.load(filepath + '/IEMOCAP/IEMOCAP_feature_valid.npy')
labels_test_ENG =  np.load(filepath + '/IEMOCAP/IEMOCAP_labels_valid.npy')


# load RECOLA input and labels to numpy arrays
samples_train_FR = np.load(filepath + '/RECOLA/RECOLA_feature_train.npy')
labels_train_FR = np.load(filepath + '/RECOLA/RECOLA_labels_train.npy')

# load RECOLA testset
samples_test_FR =  np.load(filepath + '/RECOLA/RECOLA_feature_valid.npy')
labels_test_FR =  np.load(filepath + '/RECOLA/RECOLA_labels_valid.npy')


# Define the hyperparameters for our network
ITERATIONS = 50000
BATCH_SIZE = 50
LEARN_RATE = 0.01
DISPLAY_STEP = 10

# define english-to-french ratio used during training
INPUT_FRENCH_PERCENTILE = 0.5

# train network by alternating the language in each batch?
USE_ALTERNATING_TRAINING = False

# define number of frames for each sample based on language used
n_frames_eng = int(len(samples_train_ENG)/len(labels_train_ENG))
n_frames_fr = int(len(samples_train_FR)/len(labels_train_FR))

# set the number of samples for both datasets
n_samples_ENG = len(labels_train_ENG)
n_samples_FR = len(labels_train_FR)

# set number of frames based on used language
if INPUT_FRENCH_PERCENTILE == 0:
    n_frames = n_frames_eng
elif INPUT_FRENCH_PERCENTILE == 1:
    n_frames = n_frames_fr
else:
    n_frames = int((n_frames_eng + n_frames_fr) / 2)
    
n_features = len(samples_train_ENG[0])
n_convInput = n_frames * n_features
n_classes = 4
dropout = 0.5

# 2 gateways for our data: one for sound samples & one for labels
x = tf.placeholder(tf.float32, shape=(None, n_features*n_frames), name="Input")
y = tf.placeholder(tf.float32, shape=(None, n_classes), name="Prediction")

# gateway for dropout
keep_prob =  tf.placeholder(tf.float32, name="Dropout")

# function to generate each batch
positions_ENG = np.arange(n_samples_ENG)
positions_FR = np.arange(n_samples_FR)
random.shuffle(positions_ENG)
random.shuffle(positions_FR)
current_position_ENG = 0
current_position_FR = 0
ENG_proportion = int(round((1-INPUT_FRENCH_PERCENTILE)*BATCH_SIZE, 0))
FR_proportion = int(round(INPUT_FRENCH_PERCENTILE*BATCH_SIZE, 0))

# creates new training batch
def get_next_train_batch():
    global USE_ALTERNATING_TRAINING
    if(USE_ALTERNATING_TRAINING):
        toggle_language()
    batch_samples = []
    batch_labels = []
    global current_position_ENG
    global current_position_FR
    current_position_ENG += ENG_proportion
    current_position_FR += FR_proportion
    #  check if there are enough training samples left for a batch, otherwise shuffle and start from beginning
    if current_position_ENG >= n_samples_ENG :
        random.shuffle(positions_ENG)
        current_position_ENG = ENG_proportion
    if current_position_FR >= n_samples_FR :
        random.shuffle(positions_FR)
        current_position_FR = FR_proportion
    # add english samples to the batch
    for i in range(current_position_ENG - ENG_proportion, current_position_ENG):
        batch_labels.append(labels_train_ENG[positions_ENG[i]])
        sample_frames = []
        counter = 0
        for j in range (positions_ENG[i] * n_frames_eng, positions_ENG[i] * n_frames_eng + n_frames):
            counter += 1
            if counter <= n_frames_eng:
                for k in range (0,n_features):  # add real features to batch
                    sample_frames.append(samples_train_ENG[j][k])
            else:  # add zeros in order to match a certain frames length
                for k in range (0,n_features):
                    sample_frames.append(0)
        batch_samples.append(sample_frames)
    # add french samples to the batch
    for i in range(current_position_FR - FR_proportion, current_position_FR):
        batch_labels.append(labels_train_FR[positions_FR[i]])
        sample_frames = []
        counter = 0
        for j in range (positions_FR[i] * n_frames_fr, positions_FR[i] * n_frames_fr + n_frames):
            counter += 1
            if counter <= n_frames_fr:  # add real features to batch
                for k in range (0,n_features):
                    sample_frames.append(samples_train_FR[j][k])
            else:  # add zeros in order to match a certain frames length
                for k in range (0,n_features):
                    sample_frames.append(0)
        batch_samples.append(sample_frames)
    return batch_samples, batch_labels

# if alternating-language-training is enabled, toggles language after each batch
USE_FRENCH = True

def toggle_language():
    global USE_FRENCH
    global n_frames
    global FRENCH_PERCENTILE
    global ENG_proportion
    global FR_proportion
    if(USE_FRENCH == True):
        FRENCH_PERCENTILE = 1
        USE_FRENCH = False
        ENG_proportion = int(round((1-FRENCH_PERCENTILE)*BATCH_SIZE, 0))
        FR_proportion = int(round(FRENCH_PERCENTILE*BATCH_SIZE, 0))
    else:
        FRENCH_PERCENTILE = 0
        USE_FRENCH = True
        ENG_proportion = int(round((1-FRENCH_PERCENTILE)*BATCH_SIZE, 0))
        FR_proportion = int(round(FRENCH_PERCENTILE*BATCH_SIZE, 0))#

# create english test batch
def get_test_batch_eng(samples, labels):
    batch_samples = []
    # add english test samples
    for i in range(0, len(labels)):
        sample_frames = []
        counter = 0
        for j in range (i * n_frames_eng, i * n_frames_eng + n_frames):
            counter += 1
            if counter <= n_frames_eng:
                for k in range (0,n_features):
                    sample_frames.append(samples[j][k])
            else:
                for k in range (0,n_features):
                    sample_frames.append(0)
        batch_samples.append(sample_frames)
    return batch_samples

# create french test batch
def get_test_batch_fr(samples, labels):
    batch_samples = []
    # add french test samples
    for i in range(0, len(labels)):
        sample_frames = []
        counter = 0
        for j in range (i * n_frames_fr, i * n_frames_fr + n_frames):
            counter += 1
            if counter <= n_frames_fr:
                for k in range (0,n_features):
                    sample_frames.append(samples[j][k])
            else:
                for k in range (0,n_features):
                    sample_frames.append(0)
        batch_samples.append(sample_frames)
    return batch_samples

# define convolution layer
def conv2d(x, W, b, name="Convolution"): 
    with tf.name_scope(name):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'VALID')
        act = tf.nn.relu(conv + b)
        return tf.nn.max_pool(act, ksize = [1, 30, 1, 1], strides = [1, 3, 1, 1], padding = 'SAME')

# create model
def conv_net(x, weights, biases):
    # reshape input data to [ #samples, #frames, # features, 1]
    x = tf.reshape(x, shape = [-1, n_frames, n_features, 1])
    # convolution layer with maxpooling
    act1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # reshape to one fully connected layer with inputs as a list
    act1 = tf.reshape(act1, [-1, weights['out'].get_shape().as_list()[0]])
    # apply dropout
    act1 = tf.nn.dropout(act1, keep_prob)
    
    # output, class prediciton
    out = tf.add(tf.matmul(act1, weights['out']), biases['out'])
    return out #return the classification


# create weights
weights = {
    'wc1': tf.Variable(tf.random_normal([10,n_features,1,50])),
    'out': tf.Variable(tf.random_normal([int(math.ceil((n_frames-9)/3) * 50), n_classes]))
}

# create biases
biases = {
    'bc1': tf.Variable(tf.random_normal([50])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# construct model
pred = conv_net(x, weights, biases)

# define optimizer and loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = LEARN_RATE).minimize(cost)

# evaluate model with tf.equal(predictedValue, testData)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# launch the graph
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
step = 1
# keep training until max iterations
while step * BATCH_SIZE < ITERATIONS:
    batch_x, batch_y = get_next_train_batch()
    # Run optimization (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
    if step % DISPLAY_STEP == 0:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        print("Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    step += 1
    
print("Optimization Finished! Testing Model...")


# Evaluate model with english test data
test_x = get_test_batch_eng(samples_test_ENG, labels_test_ENG)
test_y = labels_test_ENG
accuracy_eng = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("English Test Accuracy: " + str(accuracy_eng))

# Evaluate model with french test data
test_x = get_test_batch_fr(samples_test_FR, labels_test_FR)
test_y = labels_test_FR
accuracy_fr = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("French Test Accuracy: " + str(accuracy_fr))