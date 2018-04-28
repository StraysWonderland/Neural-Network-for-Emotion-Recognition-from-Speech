# Neural-Network-for-Emotion-Recognition-from-Speech

Mono- and cross-lingual emotion classification in recorded speech through a convolutional neural network.

### Data
This model was trained and tested on a collective dataset, 
consisting of the english [IEMOCAP](https://sail.usc.edu/iemocap/)
and the french [RECOLA](http://diuf.unifr.ch/diva/recola/index.html) datasets.

### Hyperparameters
Parameter| Value  
:--|--:
Activation Functions | Relu
Loss Function | Softmax Cross Entropy
Optimizer | ADAM
Init. Learning Rate | 0.001
Mini-batch size | 50
Stride | 3
Dropout | 0.5
Epoches | 50

### achieved accuracy
Englisch testset

|Class | Mono-lingual | Multi-lingual | Cross-language
|:--|:--|:--|:--|
|Sadness | 0.000 | 0.015| 0.015
|Anger |0.019 |0.014| 0.014
|Pleasure| 0.043 |0.010| 0.120
|Joy| 0.942| 0.985| 0.864
|MICRO| 0.421| 0.432| 0.405

French testset

Class | Mono-lingual | Multi-lingual | Cross-language
:--|:--|:--|:--|
Sadness| 0.070 |0.000 |0.230
Anger| 0.200 |0.200 |0.200
|Pleasure| 0.350 |0.035 |0.357
Joy| 0.754| 0.912 |0.403
MICRO| 0.533 |0.524 |0.359

## Getting Started
You can view the notebook here on github. 
### Run the notebook
#### Prerequisites
- Python 3
- Tensorflow
- Jupyter

#### starting the notebook
Simply open a new terminal in the directory and type:
```bash
> jupyter notebook
```
#### setup model
make sure you run all codeblocks from top to bottom to setup the network

### Running the tests
To test the model, you need only to run the last codeblock.
This will evaluate the model and print the accuracy for each testset.


## Built With

* [Tensorflow](https://www.tensorflow.org/) - The framework to create the model
* [Project Jupyter](https://jupyter.org/) - Nice and easy python notebooks


## Contributors

* **A. Kaplan** - [nymvno](https://github.com/nymvno)
* **F. Strohm** - [StrohmFn](https://github.com/StrohmFn)
