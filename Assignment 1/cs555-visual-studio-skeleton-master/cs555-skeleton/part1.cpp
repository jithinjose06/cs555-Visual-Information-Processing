
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int hist_values[255];
																
Mat img,img2;

void generateHistogram(Mat img, int hist_values[])
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			hist_values[int(img.at<uchar>(i, j))]++;						
		}
	}
}

void generateNegativeImage(Mat img, int hist_values[])
{
	for (int i = 0; i < img.rows; i++)														
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);									
			hist_values[int(img.at<uchar>(i, j))]++;						
		}
	}
}


Mat calculateHistogram(Mat img){
    
    int histSize = 256; 
    float range[] = {0, 255};
    const float* histRange = {range};
    Mat hist_img;
    calcHist(&img, 1, 0, Mat(), hist_img, 1, &histSize, &histRange);
    
    
    cv::Mat dst(256, 256, CV_8UC1, Scalar(255));
    float max = 0;
    for(int i=0; i<histSize; i++){
        if( max < hist_img.at<float>(i))
            max = hist_img.at<float>(i);
    }

    float scale = (0.9*256)/max;
    for(int i=0; i<histSize; i++){
        int intensity = static_cast<int>(hist_img.at<float>(i)*scale);
        line(dst,Point(i,255),Point(i,255-intensity),Scalar(0));
    }
    return dst;

}

Mat eqHist(Mat img){
    
    int flat_img[256] = {0};
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            int index;
            index = static_cast<int>(img.at<uchar>(i,j)); 
            flat_img[index]++;
        }
    }

    
    int cumulative_sum[256]={0};
    int memory=0;
    for(int i=0; i<256; i++){
        memory += flat_img[i];
        cumulative_sum[i] = memory;
    }

    // using general histogram equalization formula
    int norm[256]={0};
    for(int i=0; i<256; i++){
    	// norm(v) = round(((cdf(v) - mincdf) / (M * N) - mincdf) * (L - 1));
        norm[i] = ((cumulative_sum[i]-cumulative_sum[0])*255)/(img.rows*img.cols-cumulative_sum[0]);
        norm[i] = static_cast<int>(norm[i]);
    }

    
    Mat result(img.rows, img.cols, CV_8U);
    
    Mat_<uchar>::iterator itr_result = result.begin<uchar>(); 
    Mat_<uchar>::iterator it_begin = img.begin<uchar>(); 
    Mat_<uchar>::iterator itr_end = img.end<uchar>(); 
    
    for(; it_begin!=itr_end; it_begin++){
        int intensity_value = static_cast<int>(*it_begin); 
        *itr_result = norm[intensity_value];
        itr_result++;
    }

    return result;
}

int main(int argc, char** argv)
{
	
	
	img = imread("../House_width_times4.bmp", IMREAD_GRAYSCALE);											
	
	for (int i = 0; i < 256; i++)																	
	{
		hist_values[i] = 0;
	}
	
	imshow("Original Image", img);																
	
	// original image  histogram
	generateHistogram(img, hist_values);
	Mat hist1 = calculateHistogram(img);
	imshow("hist of original",hist1);

	Mat equalized_image = eqHist(img);
	imshow("equalized image",equalized_image);

	Mat hist3 = calculateHistogram(equalized_image);
	imshow("histogram of 1st equalized image",hist3);
	
	// Negative Image
	generateNegativeImage(img, hist_values);
	imshow("Negative", img);
	Mat hist2 = calculateHistogram(img);
	imshow("hist of negative",hist2);

	// Equalization

	

	// Equalization for 2nd Image

	img2 = imread("../NYC_width_4times.bmp", IMREAD_GRAYSCALE);
	imshow("Original Image 2",img2);
	
	Mat hist4 = calculateHistogram(img2);
	imshow("histogram before eq",hist4);
	
	Mat outImg = eqHist(img2);
	imshow("after eq",outImg);

	Mat hist5 = calculateHistogram(outImg);
	imshow("hist after eq",hist5);

	waitKey(0);
	
	return 0;
}