## Online ROS Demo

## Dependencies

- deep-lcd compiled as directed in the parent directory's README
- ROS (tested on Kinetic)

## Build

Run the catkin build script here, or just use `catkin_make`. The CMakeLists.txt already has it set to release mode.

## Run

You will need to have roscore running. This demo requires an image topic and either a PointStamped topic or a TransformStamped topic. You can toggle which the program expects by changing the `full_transform` variable to either true or false in launch/online-demo.launch. You must also change the values of the rostopics in that file depending on your dataset. 

Run with:
```
$ source devel/setup.bash
$ roslaunch launch/online-demo.launch
```
Rviz should open, and you should see many lines of caffe logging output.
After that, start up your rosbag or whatever method you use to send the messages.
