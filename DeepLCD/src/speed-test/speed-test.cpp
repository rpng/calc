#include "deeplcd.h"


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "boost/filesystem.hpp"
#include <ctime>
#include <stdlib.h>
#include <math.h>
#include <numeric>

/*************************************************************************
* This creates an executable to test both descriptor compute time and query time.
* Simply run:
* 	$ speed-test <mem dir> <live dir> <(optional) GPU_ID (default is use CPU)>
* where <mem dir> is the directory containing the images to be saved into memory,
* and <live dir> is the directory containing the images of the same locations but with 
* altered viewpoints. You could easily just throw any two directories in here, since this does not 
* test accuracy.
**************************************************************************/ 

int main(int argc, char** argv)
{

	if (argc < 3)
	{
		std::cout << "Usage:\n\tspeed-test <mem dir> <live dir> <(optional) GPU_ID (default is use CPU)> \n";
		return -1;
	}
	boost::filesystem::path mem(argv[1]);
	boost::filesystem::path live(argv[2]);
	boost::filesystem::directory_iterator end_itr; // NULL
	int gpu_id = -1;
	if (argc == 4)	
		gpu_id = atoi(argv[3]);
	
	deeplcd::DeepLCD lcd("calc_model/deploy.prototxt","calc_model/calc.caffemodel", gpu_id);
	cv::Mat im;
	cv::Size sz(160, 120);
	clock_t start;
	std::list<double> comp_t_list /* descriptor compute + database add times */, query_t_list /* descriptor compute + database query times */;
	double dt;
	size_t id;
	std::cout << "\n----------------- Database Creation -------------------------\n";

	for (boost::filesystem::directory_iterator itr1(mem); itr1!=end_itr; ++itr1)
	{	
		std::cout << "loading database image from: " << itr1->path().string() << "\n";
		im = cv::imread(itr1->path().string(), cv::IMREAD_GRAYSCALE);
		cv::resize(im, im, sz);
		start = clock();
		id = lcd.add(im);
		dt = ((double)clock()-start)/CLOCKS_PER_SEC;
		std::cout << "Time from image to DB = " << dt*1000 << " ms\n";
		comp_t_list.push_front(dt);  
		std::cout << "Added Image " << id << " to database\n\n";
	}
	
	std::cout << "\n----------------- Database Query -------------------------\n";

	// Okay now we have a database of descriptors, lets time the querie
	deeplcd::query_result q; // This is unused since we only do a speed test
	for (boost::filesystem::directory_iterator itr2(live); itr2!=end_itr; ++itr2)
	{	
		std::cout << "loading query image from: " << itr2->path().string() << "\n";
		im = cv::imread(itr2->path().string(), cv::IMREAD_GRAYSCALE);
		cv::resize(im, im, sz); 
		const deeplcd::descriptor d = lcd.calcDescr(im);
		start = clock();
		q = lcd.query(d, 0); // query(descriptor, false) means return 1 result in q, and don't add im's descriptor to database afterwards
		dt = ((double)clock()-start)/CLOCKS_PER_SEC;
		query_t_list.push_front(dt);  
		std::cout << q << "\n";
		std::cout << "Database query time = " << dt*1000 << " ms\n\n";
	}

	std::cout << "\n\n\t\t\tResults\n\t\t\t-------\n\n";
	std::cout << "Time from image to DB:\n\t\t\t";
	double mean = std::accumulate(comp_t_list.begin(), comp_t_list.end(), 0.0) / (double)comp_t_list.size();

	double stdev = 0.0;	
	for (double x : comp_t_list)
		stdev += pow(x-mean, 2); 
	stdev = sqrt( stdev / ((double)comp_t_list.size()-1.0) );
	std::cout << "Extraction Time Mean: " << mean*1000 << ", StDev: " << stdev*1000 << " (ms) \n";

	std::cout << "Database querying time for a database of size " << query_t_list.size() << ":\n\t\t\t";
	mean = std::accumulate(query_t_list.begin(), query_t_list.end(), 0.0) / (double)query_t_list.size();
	stdev = 0.0;
	for (double x : query_t_list)
		stdev += pow(x-mean, 2); 
	stdev = sqrt( stdev / ((double)query_t_list.size()-1.0) );
	std::cout << "Query Time Mean: " << mean*1000 << ", StDev: " << stdev*1000 << " (ms) \n";
} 


