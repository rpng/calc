# Train and Test

This module provides python code to create lmdb databases for CALC training, train the net, and then test multiple nets against AlexNet, DBoW2, and HOG. Note that the HOG used here has the same parameters as used in CALC training. HOG with default parameters is much better, but we wanted to show the power of convolutional autoencoders to learn useful information from not-so-useful data. Also note that single gpu only training is available here due to pycaffe limitations. To train on multiple gpus, we simply copied the caffe executable into this directory and ran it from there.

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
- cython (HIGHLY recommended for speedup)
- ThirdParty/DBoW2 (see below for details)

Required for DBoW2 installation:
- Boost and Boost python

## Cloning this repo (and optionally the DBoW2 submodule)

If you want to see precision-recall against DBoW2, you will have to use my fork of DBoW2, which supplies limited python wrappers (nothing actually useful for SLAM). To do this run `git clone --recursive`, since it is set up as a submodule. Otherwise just run `git clone`, and DBoW2 will not be cloned. testNet is set up to only import DBoW2 and add it to plots if the library is found compiled.

If you want to compile DBoW2, and you have cloned this repo recursively, see ThirdParty/DBoW2/README.md for the instructions.

## writeDatabase

This library is for creating the lmdb databases for CALC training. All you need is a directory tree containing images, and the program will recursively seek out all the image files in the tree.

## makeNet

This library defines the CALC net and solver files, and can also be used to train it, even though multi-gpu is not supported.

## testNet

This library is for generating precision-recall curves, comparing against HOG (as described above), and optionally DBoW2 and AlexNet. It will also tell you the max score threshold that keeps the precision at 1.0. This will be extremely useful when using the DeepLCD library for loop detection. For example, from the Gardens Point walking dataset, comparing left vs right, I found an optimal threshold to be 0.91. You can see how this value varies between datasets in order to choose one for your SLAM system. 

## Cythonize python modules

I added a Makefile to cythonize all of the python to make it a bit faster, especially since writeDatabase and testNet consist of long-running loops. If cython and build-essential is installed, simply run `make` in this dirtectory to cythonize all the python libraries, then run the code from main.py

## main.py

A simple main for writeDatabase, makeNet, and testNet, as well as some extra functionalities, where all the magic happens. Note that this will work whether the other libraries are cythonized or not. There are many options to branch from this main--listed below. Mains are not provided in the other python files in order to keep the argument parsing in one place and as simple as possible. You can see`./main.py <arg> -h` for help in that specific area.

```
usage: main.py [-h] {net,db,view,plot,test} ...

positional arguments:
  {net,db,view,plot,test}
    net                 Allows access to functions in makeNet.py, which allow
                        for net definition, training, and descriptor
                        dimension viewing. Run `./main.py net -h` to see all
                        options
    db                  Allows access to functions in writeDatabase.py, which
                        allow for database writing from raw image files,
                        database testing, and optional training when the
                        writing is done (to save yourself a trip). Run
                        `./main.py db -h` to see all options
    view                View an example of the altered image copies
    plot                Compare varying database times between DBoW2 and
                        DeepLCD
    test                Allows for testing net(s) against some optional
                        benchmark algorithms with precision-recall curves, and
                        timing. Also allows for viewing thresholds used in
                        those curves. Run `./main.py test -h` to see all
                        options

optional arguments:
  -h, --help            show this help message and exit
```




