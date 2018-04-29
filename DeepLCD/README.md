# DeepLCD

A C++ library for online SLAM loop closure, using CALC models. 
Included with the shared library is our pre-trained model (see get_model.sh), which is downloaded on compilation, a useful demo, unit testing (CPU tests only for now), and a speed test, as well as an online loop closure demo with ROS!

## Dependencies

Required:
- OpenCV >= 2.0
- Eigen >= 3.0
- Boost filesystem
- Caffe 

Optional but HIGHLY Recommended:
- CUDA

## To Compile

```
$ mkdir build && cd build
$ cmake .. && make # Already set to Release build
```

Note that if your caffe is not installed in ~/caffe, you must use 

```
$ cmake -DCaffe_ROOT_DIR=</path/to/caffe> .. && make
```
instead.

## To Run Tests

```
$ cd /path/to/build
$ ./deeplcd-test
```

Start an issue if tests fail!!!

## To Run the Demo

The demo uses 6 images that are included in this repo. They are from the [KITTI VO dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). 
They are stereo pairs located in src/images/live and src/images/memory.
 Each image in live/memory with the same file name is from the same point in time.
 The demo first loads the memory images into a database, then performs a search with the live images.
 The default model downloaded by get_model.sh is used here, and caffe is set to cpu. 
 You should see matching image IDs in the printed output.
 That means that the search successfully matched the memory images to the live ones.
 This demo is useful for seeing how to implement your own code with this library, whether it be for an image retrieval task, or a SLAM system. 

Not: See the overloads of DeepLCD::query in include/deeplcd/deeplcd.h.
 The demo uses the optimized k-nearest-neighbors search that only works for 1 neighbor (because the return value doesn't need to be wrapped in an std::vector). However, there are overloads that allow for k neighbors to be returned (in order of closest to farthest). So use the overload if you want k > 1.

## To Run the Speed Test

```
$ speed-test <mem dir> <live dir> <(optional) GPU_ID (default=-1 for cpu)>
```

where mem/live dir are directories containing images to use in the test. For example, if you have two directories for left/right stereo pairs, you can throw those in the arguments. GPU_ID defaults to -1, which means use the CPU, but just use the number of the desired GPU to run it on that device.


## Online Loop Closure Demo with ROS

See online-demo-ws
