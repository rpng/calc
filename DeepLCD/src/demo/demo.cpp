#include "deeplcd.h"


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

/*************************************************************************
* Here is quick cpu demo using a few images from the Kitti odometry dataset
**************************************************************************/ 

int main(int argc, char** argv)
{
	
	std::cout << "Deep Loop Closure Detection Demo!\n";

	std::vector<std::string> fls = {"000000.png", "001000.png", "002000.png"};
	std::string images = "images/";
	std::string live = "live/";
	std::string mem = "memory/";

	deeplcd::DeepLCD test_lcd; // Using default constructor, takes net from rvbaa_model directory downloaded on compilation

	cv::Mat im;

	cv::Size sz(160, 120);
	
	std::string curr;
	
	for (std::string fl : fls)
	{	
		curr = images + mem + fl;
		std::cout << "Loading image " << curr << "\n";
		im = cv::imread(curr);
		cv::cvtColor(im, im, cv::COLOR_BGR2GRAY); // convert to grayscale
		cv::resize(im, im, sz);
		std::cout << "Added Image " << test_lcd.add(im) << " to database\n";
	}
	
	std::cout << "\n------------------------------------------\n";

	// Okay now we have a database of descriptors, lets see if we can match them now
	int i = 0;
	deeplcd::query_result q;
	for (std::string fl : fls)
	{	
		curr = images + live + fl;
		std::cout << "\nLoading image " << curr << " for database query\n";
		im = cv::imread(curr);
		cv::cvtColor(im, im, cv::COLOR_BGR2GRAY); // convert to grayscale
		cv::resize(im, im, sz);
		q = test_lcd.query(im, 0); // query(im, false) means return 1 result in q, and don't add im's descriptor to database afterwards
		std::cout << "Image " << i++ << " result: " << q << "\n";
	}

} 


