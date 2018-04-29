#include "deeplcd.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "boost/filesystem.hpp"
#include <ctime>
#include <stdlib.h>
#include <fstream>
#include <iterator>
#include <math.h>

/*************************************************************************
* This creates an executable to test query time for varying database size.
* Simply run:
* 	$ vary-db-size <mem dir> <live dir> <(optional) GPU_ID (default is use CPU)>
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
	size_t id;
	std::list<deeplcd::descriptor> query_des;

	// calculate query descriptors just once	
        for (boost::filesystem::directory_iterator itr2(live); itr2!=end_itr; ++itr2) 
	{
		im = cv::imread(itr2->path().string(), cv::IMREAD_GRAYSCALE);
		cv::resize(im, im, sz); 
                query_des.push_front(lcd.calcDescr(im));
	}

	size_t m = 1, q = query_des.size(), n = 0;
	double query_t = 0.0;
	std::vector<double> times; // results 
	std::vector<size_t> db_sizes; // results 
	
	for (boost::filesystem::directory_iterator itr1(mem); itr1!=end_itr; ++itr1)
	{
		n++;
		im = cv::imread(itr1->path().string(), cv::IMREAD_GRAYSCALE);
		cv::resize(im, im, sz);
		id = lcd.add(im);
		std::cout << "Added Image " << id << " to database\n\n";
	
		if (m%100==0)
		{
	                std::cout << "Timing queries for database size=" << m << "\n";
			for (deeplcd::descriptor d : query_des)
			{	
				start = clock();
				lcd.query(d, 0); // query(descriptor, false) means return 1 result in q, and don't add im's descriptor to database afterwards
				query_t += ((double)clock()-start)/CLOCKS_PER_SEC * 1000 ; // (ms)
			}
			times.push_back(query_t / q);
			db_sizes.push_back(m);
			query_t = 0.0;
		}	
		m++;
	}
	
	std::cout << "\n\n\tResults\n\n";
	std::cout << "bins = " << n / 10 << "\n";
	std::cout << "N = " << n << "\n";
	
	times.shrink_to_fit();
	db_sizes.shrink_to_fit();

	std::string f = "vary-db-size-calc-results.txt";
	std::ofstream results_file(f);
	std::ostream_iterator<size_t> m_it(results_file, " ");
	std::ostream_iterator<double> t_it(results_file, " ");
	std::copy(db_sizes.begin(), db_sizes.end(), m_it);
	results_file << "\n";
	std::copy(times.begin(), times.end(), t_it);
	std::cout << "results written to " << f << "\n";
} 


