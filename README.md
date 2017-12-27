# SiameseCBOW

Implementation of [SiameseCBOW](http://www.aclweb.org/anthology/P16-1089) using python3.4, keras and tensorflow.

## Environments

* Mac OS 10.10.5
* python 3.4.3
* Keras 2.1.2
* tensorflow 1.4.0
  * Mac, CPU version:https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.0-py3-none-any.whl

## How to use

### 0. Installation

```
$ pip install -r requirments.txt
```

### 1. Customize src/load.py

At first, you should customize src/load.py so that you can load your data and preprocess it.

If you just want to test the codes, please comment out ```x, y = load(file)``` and set variable:x, y in main.py like this:

```
def main():

...

 import numpy as np
 x = [np.ones((10, input_length))] * (1 + n_positive + n_negative)
 y = np.ones((10, n_positive + n_negative))

...

 model.fit(x, y, epochs=1)
```

### 2. Set Hyper-parameters

If you want, please set Hyper-parameters such as embedding dimension in main.py.

### 3. Train&Save

Please execute this command at the git project directory:

```
$ python main.py -f <data_path>
```

and a pickle file of an embedding vector will be saved in ```./save/```.
