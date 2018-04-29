## Online ROS Demo

## Dependencies

- deep-lcd compiled as directed in the parent directory's README
- ROS (tested on Kinetic)

## Build

Run the catkin build script here, or just use `catkin_make`.

## Run

You will need to have roscore running. This demo requires an image topic and either a PointStamped topic or a TransformStamped topic. You can toggle which the program expects by changing the `full_transform` variable to either true or false in launch/online-demo.launch. You must also change the values of the rostopics in that file depending on your dataset. 
