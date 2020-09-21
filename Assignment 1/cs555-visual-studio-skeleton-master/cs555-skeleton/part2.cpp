
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include<math.h>


using namespace cv;
using namespace std;



Mat binaryImageGenerator(Mat img, int threshold)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (int(img.at<uchar>(i, j)) >= threshold)
			{
				img.at<uchar>(i, j) = 255;																						
			}
			else
			{
				img.at<uchar>(i, j) = 0;																						
			}
		}
	}

	return img;
}

int averageCalculate(int hist[], int x, int y)													
{
	int sum=0;
	sum = sum + hist[0];
	int productSum = (hist[0] * 0);

	for (int i = x; i < y; i++)
	{
		sum = sum + hist[i];
	}

	for (int i = x; i < y; i++)
	{
		productSum = productSum+ (hist[i] * i) ;
	}

	int average = productSum / sum;	

	return average;
}

void binaryImage(Mat img,const char* s,const char* s1)
{
	imshow(s, img);
	int rows = img.rows;
	int cols = img.cols;
	int hist_values[255];
	for (int a = 0; a < 256; a++)																			
	{
		hist_values[a] = 0;
	}

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			hist_values[int(img.at<uchar>(i, j))]++;										
		}
	}

	int old_threshold = 128;
	int new_threshold = 128;
	int diff_threshold = 2;
	int avg1, avg2;
	
	do
	{
		old_threshold = new_threshold;
		avg1 = averageCalculate(hist_values, 1, old_threshold);
		avg2 = averageCalculate(hist_values, old_threshold + 1, 256);
		new_threshold = (avg1 + avg2) / 2;
		
	}while ((old_threshold-new_threshold) > diff_threshold);

	img = binaryImageGenerator(img, new_threshold);
	cout << "Threshold: " << new_threshold << endl;
	imshow(s1, img);
	
	
}

int main(int argc, char** argv)
{

	

	Mat img = imread("../shapes2.1.bmp", IMREAD_GRAYSCALE);
	Mat img1 = imread("../guide_8bits.bmp", IMREAD_GRAYSCALE);
	binaryImage(img,"image 1","binaryImage 1");
	binaryImage(img1,"image 2","binaryImage 2");
	waitKey(0);													

	return 0;
}
