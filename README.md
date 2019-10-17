```
   ******      **     **         ****** 
  **////**    ****   /**        **////**
 **    //    **//**  /**       **    // 
/**         **  //** /**      /**       
/**        **********/**      /**       
//**    **/**//////**/**      //**    **
 //****** /**     /**/******** //****** 
  //////  //      // ////////   //////
```


[![Build Status](https://travis-ci.org/rpng/calc.svg?branch=master)](https://travis-ci.org/rpng/calc)

Convolutional Autoencoder for Loop Closure. A fast deep learning architecture for robust SLAM loop closure, or any other place recognition tasks. Download our pre-trained model with `./DeepLCD/get_model.sh`, or train your own!

This repo is separated into two modules. TrainAndTest for training and testing models, and DeepLCD, which is a C++ library for online loop closure or image retrieval. See their respective READMEs for details.  

If you use this code in any publication, please cite our paper. The pdf can be found [here](http://www.roboticsproceedings.org/rss14/p32.pdf), and the reference should be in this format:

```
@InProceedings{Merrill2018RSS,
  Title                    = {Lightweight Unsupervised Deep Loop Closure},
  Author                   = {Nathaniel Merrill and Guoquan Huang},
  Booktitle                = {Proc. of Robotics: Science and Systems (RSS)},
  Year                     = {2018},
  Address                  = {Pittsburgh, PA},
  Month                    = jun # { 26-30, }
}
```
