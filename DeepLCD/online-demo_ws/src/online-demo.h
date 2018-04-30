#include "deeplcd.h"
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <cstring>
#include <list>

deeplcd::DeepLCD* lcd;
ros::Publisher marker_pub;
image_transport::Publisher im_pub;
uint32_t i = 1;
visualization_msgs::Marker line_strip, loop_lines;
std::vector<geometry_msgs::Point> db_points;
std::vector<cv::Mat> kf;
std::vector<int> loop_ids;
int loop_hyp_cnt = 0;
long int last_loop_hyp_id = -1;
int min_loop_hyp_cnt;
double thresh;

bool detect_loop(const cv::Mat& im);
void mono_callback(const sensor_msgs::ImageConstPtr& msg);
void transform_callback(const geometry_msgs::TransformStampedConstPtr& msg);
void point_callback(const geometry_msgs::PointStampedConstPtr& msg);

