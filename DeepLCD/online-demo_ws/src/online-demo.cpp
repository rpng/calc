#include "online-demo.h"
#include <stdlib.h>
#include "opencv2/highgui.hpp"

bool detect_loop(const cv::Mat& im)
{
	deeplcd::query_result q = lcd->query(im);
	bool loop_closed = 0;
	if (q.score > thresh && std::abs((int)db_points.size()-(int)q.id) > 10)
	{
		loop_hyp_cnt++;
		std::cout << "Loop hypothesis strengthened! Curr frame = " << db_points.size() << ", " << q << "\n";
		if (loop_hyp_cnt == min_loop_hyp_cnt)
		{
			loop_lines.points.push_back(db_points[db_points.size()-min_loop_hyp_cnt/2]); // current point for display (From the middle of the loop hypothesis)
			loop_lines.points.push_back(db_points[q.id]); // loop point for display
			loop_closed = 1;
			std::cout << "Loop Closed! Curr frame = " << db_points.size() << ", " << q << "\n";
			cv::Size sz = im.size();
			cv::Mat im_match(sz.height, 2*sz.width, im.type());
			cv::Mat left(im_match, cv::Rect(0, 0, sz.width, sz.height));
		    	cv::Mat right(im_match, cv::Rect(sz.width, 0, sz.width, sz.height));
			im.copyTo(left);
			kf[q.id].copyTo(right);
			//cv::imshow("bleh", im_match);
			//cv::waitKey(0);
			sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", im_match).toImageMsg();
			im_pub.publish(msg);
			loop_hyp_cnt = 0;
			
		}
	}
	else
	{
		loop_hyp_cnt = 0;
	}
	return loop_closed;
}
void mono_callback(const sensor_msgs::ImageConstPtr& msg)
{
	cv::Mat im;
	try {
		// Copy the ros image message to cv::Mat.
		cv_bridge::CvImageConstPtr image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		// Convert to gray scale
		if(image->image.channels() == 3)
		    cv::cvtColor(image->image, im, CV_BGR2GRAY);
		if(image->image.channels() == 4)
		    cv::cvtColor(image->image, im, CV_RGBA2GRAY);
		else
		    cv::cvtColor(image->image, im, CV_BGR2GRAY);
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	if ((i++ % 7) == 0) // mimic adding KF every 7frames (Totally fake)
	{
		if (line_strip.points.size() > 0) 
		{
			db_points.push_back(line_strip.points.back());// These are in lieu of keyframes
			cv::resize(im, im, cv::Size_<uint8_t>(160, 120));
			kf.push_back(im);
			deeplcd::query_result q;
			detect_loop(im);
		}
	}
	if (loop_lines.points.size() > 0) marker_pub.publish(loop_lines);
	//std::cout << "Image\n" << i++ << "\n";
}

void transform_callback(const geometry_msgs::TransformStampedConstPtr& msg)
{
	geometry_msgs::Point p;
	p.x = msg->transform.translation.x;
	p.y = msg->transform.translation.y;
	p.z = msg->transform.translation.z;
	//printf("\nPoint:\n x:%3.3f y:%3.3f z:%3.3f\n", p.x, p.y, p.z); 
      	line_strip.points.push_back(p);
    	marker_pub.publish(line_strip);
}

void point_callback(const geometry_msgs::PointStampedConstPtr& msg)
{
	//printf("\nPoint:\n x:%3.3f y:%3.3f z:%3.3f\n", msg->point.x, msg->point.y, msg->point.z); 
	geometry_msgs::Point p;
	p.x = msg->point.x; p.y = msg->point.y; p.z = msg->point.z;
      	line_strip.points.push_back(p);
    	marker_pub.publish(line_strip);
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "deeplcd_online");
	ros::NodeHandle nh("~");
	std::string net_def, weights, pos_topic, im_topic;
	int gpu_id;
	bool full_transform;
	nh.param<std::string>("net_def", net_def, "devel/calc_model/deploy.prototxt");
	nh.param<std::string>("weights", weights, "devel/calc_model/calc.caffemodel");
	nh.param<std::string>("pos_topic", pos_topic, "/leica/position");
	nh.param<std::string>("im_topic", im_topic, "/cam0/image_raw");
	nh.param<int>("gpu", gpu_id, -1);
	nh.param<int>("min_loop_hyp_cnt", min_loop_hyp_cnt, 3);
	nh.param<float>("base_thresh", thresh, 0.9);
	nh.param<bool>("full_transform", full_transform, 0);
	std::cout << "\n\nnet def file: " << net_def << "\nweights: " << weights << "\n";
	std::cout << "im_topic: " << im_topic << "\npos_topic: " << pos_topic << "\n";
	printf("gpu: %d\nbase_thresh: %1.1f\nfull_transform: %d\nmin_loop_hyp_cnt: %d", gpu_id, thresh, full_transform, min_loop_hyp_cnt);
	lcd = new deeplcd::DeepLCD(net_def, weights, gpu_id);
	ros::Subscriber im_sub = nh.subscribe(im_topic, 10, mono_callback);
	ros::Subscriber pos_sub;
	if (full_transform) pos_sub = nh.subscribe(pos_topic, 100, transform_callback);
	else pos_sub = nh.subscribe(pos_topic, 100, point_callback);
	marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

    	loop_lines.header.frame_id = line_strip.header.frame_id  = "/base_link";
   	loop_lines.header.stamp = line_strip.header.stamp = ros::Time::now();
    	loop_lines.ns = line_strip.ns = "points_and_lines";
    	loop_lines.action = line_strip.action = visualization_msgs::Marker::ADD;
    	loop_lines.pose.orientation.w = line_strip.pose.orientation.w = 1.0;

   	line_strip.id = 0;
    	loop_lines.id = 1;

    	line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    	loop_lines.type = visualization_msgs::Marker::LINE_LIST;
  

    	// LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
    	line_strip.scale.x = loop_lines.scale.x = 0.1;

    	// Line strip is blue
    	line_strip.color.b = 1.0;
    	line_strip.color.a = 1.0;
 
   	// Loop lines are green
    	loop_lines.color.g = 1.0;
    	loop_lines.color.a = 1.0;

  	image_transport::ImageTransport it(nh);
	im_pub = it.advertise("im_match", 1);	
	
	ros::spin();

	return 0;
}
