#include "gtest/gtest.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "deeplcd.h"

std::vector<std::string> fls = {"000000.png", "001000.png", "002000.png"};
std::string images = "images/";
std::string live = "live/";
std::string mem = "memory/";

// This model will be downloaded upon compilation
std::string model_dir = "calc_model/";
std::string net_def = model_dir + "deploy.prototxt";
std::string net_weights = model_dir + "calc.caffemodel";

std::string curr;

deeplcd::DeepLCD* test_lcd;

deeplcd::QueryResults q;

cv::Mat im;

cv::Size sz(160, 120);

cv::Mat load_and_test(std::string curr)
{
	im = cv::imread(curr);
	EXPECT_TRUE(im.data) << "Image " << curr << " could not be loaded! Please check that it exists\n";
	cv::cvtColor(im, im, cv::COLOR_BGR2GRAY); // convert to grayscale
	cv::resize(im, im, sz);
	return im;
}


TEST(DeepLCDTest, TestDefaultConstructor)
{
	test_lcd = new deeplcd::DeepLCD();
	EXPECT_TRUE(test_lcd) << "Default constructor created a NULL pointer\n";
	EXPECT_TRUE(test_lcd->autoencoder) << "Default constructor could not load net\n";
}

TEST(DeepLCDTest, TestAdd)
{
	size_t i = 0;
	for (std::string fl : fls)
	{	
		curr = images + mem + fl;
		im = load_and_test(curr);	
		test_lcd->add(im);
		EXPECT_EQ( test_lcd->db.size(), ++i) << "Image descriptor was not added to database correctly\n";
	}

} 

TEST(DeepLCDTest, TestImageRetrieval)
{
	size_t i = 0;
	for (std::string fl : fls)
	{
		curr = images + live + fl;
		im = load_and_test(curr);	
		test_lcd->query(im, q, 1, 0); 
		EXPECT_EQ( q[0].id, i++) << "Query result with max_res=1 is wrong!";
	}
}


TEST(DeepLCDTest, TestFullConstructor)
{
	delete test_lcd;
	test_lcd = new deeplcd::DeepLCD(net_def, net_weights, -1);
	EXPECT_TRUE(test_lcd) << "Full constructor created a NULL pointer\n";
	EXPECT_TRUE(test_lcd->autoencoder) << "Full constructor could not load net\n";
}

TEST(DeepLCDTest, TestAddMulti)
{
	size_t i = 0;
	for (std::string fl : fls)
	{	
		curr = images + mem + fl;
		im = load_and_test(curr);	
		test_lcd->add(im);
		EXPECT_EQ( test_lcd->db.size(), ++i) << "Image descriptor was not added to database correctly\n";
	}

} 


TEST(DeepLCDTest, TestMultiImageRetrieval)
{
	size_t i = 0;
	for (std::string fl : fls)
	{
		curr = images + live + fl;
		im = load_and_test(curr);	
		test_lcd->query(im, q, 3, 0); 
		EXPECT_EQ( q[0].id, i++) << "Query result with max_res=3 is wrong!";
		EXPECT_EQ( q.size(), 3) << "QueryResult object is only size 3 when max_res==3!!";
	}
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
