# Train and Test

This module provides python code to create lmdb databases for CALC training, train the net, and then test multiple nets against AlexNet, DBoW2, and HOG. Note that the HOG used here has the same parameters as used in CALC training. HOG with default parameters is much better, but we wanted to show the power of convolutional autoencoders to learn useful information from not-so-useful data. Also note that single gpu only training is available here due to pycaffe limitations. To train on multiple gpus, use `train_multiple_gpu.sh` after creating the databases and defining a net.

## Dependencies

Required for training and database writing:
- python2.7 
- OpenCV
- lmdb
- numpy
- pycaffe
- argparse

Required for testing:
- sklearn
- matplotlib

Optional:
- build-essential (for Makefile)
- cython
- ThirdParty/DBoW2 (see below for details)

Required for DBoW2 installation:
- Boost and Boost python

## Cloning this repo (and optionally the DBoW2 submodule)

If you want to see precision-recall against DBoW2, you will have to use my fork of DBoW2, which supplies limited python wrappers (nothing actually useful for SLAM). To do this run `git clone --recursive`, since it is set up as a submodule. Otherwise just run `git clone`, and DBoW2 will not be cloned. testNet is set up to only import DBoW2 and add it to plots if the library is found compiled.

If you want to compile DBoW2, and you have cloned this repo recursively, see ThirdParty/DBoW2/README.md for the instructions.

## writeDatabase

This library is for creating the lmdb databases for CALC training. All you need is a directory tree containing images, and the program will recursively seek out all the image files in the tree.

## makeNet

This library defines the CALC net and solver files, and can also be used to train it, even though multi-gpu is not supported (use `train_multi_gpu.sh` for that).

## testNet

This library is for generating precision-recall curves, comparing against HOG (as described above), and optionally DBoW2 and AlexNet. It will also tell you the minimum score threshold that keeps the precision at 1.0 (argmax(recall(threshold)) or -1.0 if precision is never 1.0). This will be extremely useful when using the DeepLCD library for loop detection. For example, from the Gardens Point walking dataset, comparing day left vs night right with `-n9`, I found an optimal threshold to be 0.9. You can see how this value varies between datasets in order to choose one for your SLAM system. 

We have provided our own small sample dataset for testing under test_data. It is called the Campus Loop dataset, and provides two sequences of images from a snowy day and a sunny day with large variations in viewpoint. It is the default dataset location for testNet, but you must unpack the tar file first.

## Cythonize python modules

I added a Makefile to cythonize all of the python to make it a bit faster, especially since writeDatabase and testNet consist of long-running loops. If cython and build-essential is installed, simply run `make` in this dirtectory to cythonize all the python libraries, then run the code from main.py

## main.py

A simple main for writeDatabase, makeNet, and testNet, as well as some extra functionalities, where all the magic happens. Note that this will work whether the other libraries are cythonized or not. There are many options to branch from this main. Mains are not provided in the other python files in order to keep the argument parsing in one place and as simple as possible. You can see`./main.py <arg> -h` for help in that specific area.





