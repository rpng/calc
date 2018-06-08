#include "online-demo.h"
#include <stdlib.h>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

bool detect_loop(const cv::Mat& im)
{
	deeplcd::query_result q = lcd->query(im, 20);

	if (q.score > (float)thresh) 
	{
		if (loop_hyp_cnt == 0)
			last_loop_hyp_id = q.id;

		loop_hyp_cnt++;
		std::cout << "Loop hypothesis strengthened! Curr frame = " << db_points.size()-1 << ", " << q << ", loop_hyp_cnt=" << loop_hyp_cnt <<"\n";

		if (std::abs(q.id - last_loop_hyp_id) < 2 * min_loop_hyp_cnt) // Check that we are looking at a query from the same general location
		{	
			if (loop_hyp_cnt == min_loop_hyp_cnt)
			{
				int mid_frame_id = db_points.size() - 1 - min_loop_hyp_cnt/2; // Index for the middle of the loop hypothesis process
				loop_lines.points.push_back(db_points[mid_frame_id]); 
				loop_lines.points.push_back(db_points[q.id]); // loop point for display
				marker_pub.publish(loop_lines);
				std::cout << "Loop Closed! Curr frame = " << db_points.size()-1 << ", " << q << "\n";
				loop_ids.push_back(mid_frame_id);
				loop_ids.push_back(q.id);
				cv::Mat curr_im = kf.back(); // Original size image for display
				cv::Size sz = curr_im.size();
				cv::Mat im_match(sz.height, 2*sz.width, curr_im.type());
				cv::Mat left(im_match, cv::Rect(0, 0, sz.width, sz.height));
				cv::Mat right(im_match, cv::Rect(sz.width, 0, sz.width, sz.height));
				curr_im.copyTo(left);
				kf[q.id].copyTo(right);
				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", im_match).toImageMsg();
				im_pub.publish(msg);
				loop_hyp_cnt = 0;	
				return 1;
			}	
		}
		else
		{		
			loop_hyp_cnt = 0;
		}
	}
	else 
	{
		loop_hyp_cnt = 0;
	}
	return 0;
}
void mono_callback(const sensor_msgs::ImageConstPtr& msg)
{
	cv::Mat im;
	try {
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
	if ((i++ % 7) == 0) // mimic adding KF every 7 frames (Totally fake)
	{
		if (line_strip.points.size() > 0) 
		{
			db_points.push_back(line_strip.points.back());// These are in lieu of keyframes
			cv::Mat imcp;
			im.copyTo(imcp);
			kf.push_back(imcp);
			
			cv::resize(im, im, cv::Size_<uint8_t>(160, 120));
			cv::equalizeHist(im, im);
			if (lcd->db.size() > lcd->n_exclude)
				detect_loop(im);
			else
				lcd->add(im);
		}
	}
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
	nh.param<double>("base_thresh", thresh, 0.9);
	nh.param<bool>("full_transform", full_transform, 0);
	std::cout << "\n\nnet def file: " << net_def << "\nweights: " << weights << "\n";
	std::cout << "im_topic: " << im_topic << "\npos_topic: " << pos_topic << "\n";
	printf("gpu: %d\nbase_thresh: %.3f\nfull_transform: %d\nmin_loop_hyp_cnt: %d", gpu_id, thresh, full_transform, min_loop_hyp_cnt);
	lcd = new deeplcd::DeepLCD(net_def, weights, gpu_id);
	nh.param<int>("n_exclude", lcd->n_exclude, 30);
	ros::Subscriber im_sub = nh.subscribe(im_topic, 10, mono_callback);
	ros::Subscriber pos_sub;

	// We can get points either from a transform topic or point topic
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
		
	// Write resulting VO points and point matches to disk
	FILE* db_fp = fopen("db-points.csv", "w");
	FILE* loop_fp = fopen("loop-points.csv", "w");
	uint32_t i;
	if (db_fp) 
	{
		fprintf(db_fp, "frame_id, x, y, z\n");
		i = 0;
		for (geometry_msgs::Point p : db_points)
			fprintf(db_fp, "%u, %f, %f, %f\n", i++, p.x, p.y, p.z);
		fclose(db_fp);
	}
	if (loop_fp)
	{
		fprintf(loop_fp, "frame_id1, frame_id2, x1, y1, z1, x2, y2, z2\n"); // The lines need two points 
		geometry_msgs::Point p;
		for (i = 0; i < loop_lines.points.size()-1; i+=2)
		{
			p = loop_lines.points[i];
			fprintf(loop_fp, "%u, %u, %f, %f, %f", loop_ids[i], loop_ids[i+1], p.x, p.y, p.z);
			p = loop_lines.points[i+1];
			fprintf(loop_fp, ", %f, %f, %f\n", p.x, p.y, p.z);
		}
		fclose(loop_fp);
	}

	return 0;
}
