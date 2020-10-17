#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int xGradient(Mat img,int x,int y)
{
	return (img.at<uchar>(x-1,y-1) + 2*img.at<uchar>(x,y-1) + img.at<uchar>(x+1,y-1) - img.at<uchar>(x-1,y+1) - 2*img.at<uchar>(x,y+1) - img.at<uchar>(x+1,y+1) );
}

int yGradient(Mat img,int x,int y)
{
	return (img.at<uchar>(x-1,y-1) + 2*img.at<uchar>(x-1,y) + img.at<uchar>(x-1,y+1) - img.at<uchar>(x+1,y-1) - 2*img.at<uchar>(x+1,y) - img.at<uchar>(x+1,y+1) );
}

void sobel(Mat img,Mat imgClone,string const &str)
{

	int gx,gy,gfinal;

	for(int i=0;i<imgClone.rows;i++)
	{
		for(int j=0;j<imgClone.cols;j++)
		{
			imgClone.at<uchar>(i,j)=0.0;
		}
	}

	for(int i=1;i<imgClone.rows-1;i++)
	{
		for(int j=1;j<imgClone.cols-1;j++)
		{	
			gx = xGradient(img,i,j);
			gy = yGradient(img,i,j);
			gfinal = abs(gx) + abs(gy);

			if(gfinal>255)
				gfinal = 255;
			if(gfinal<0)
				gfinal = 0;

			imgClone.at<uchar>(i,j) = gfinal;
		}
	}

	imshow(str,imgClone);

	

}
void unsharpMask(Mat img,Mat imgClone,string const &str,string const &str1)
{
	float mask[3][3] = {{0.0625,0.125,0.0625},
				  {0.125,0.25,0.125},
				  {0.0625,0.125,0.0625}};

	for(int i=0;i<imgClone.rows;i++)
	{
		for(int j=0;j<imgClone.cols;j++)
		{
			imgClone.at<uchar>(i,j)=0.0;
		}
	}

	for(int i=1;i<imgClone.rows-1;i++)
	{
		for(int j=1;j<imgClone.cols-1;j++)
		{
			imgClone.at<uchar>(i,j)=img.at<uchar>(i-1,j-1)*mask[0][0] + img.at<uchar>(i-1,j)*mask[0][1] + img.at<uchar>(i-1,j+1)*mask[0][2] + img.at<uchar>(i,j-1)*mask[1][0] + img.at<uchar>(i,j)*mask[1][1] + img.at<uchar>(i,j+1)*mask[1][2] + img.at<uchar>(i+1,j-1)*mask[2][0] + img.at<uchar>(i+1,j)*mask[2][1] + img.at<uchar>(i+1,j+1)*mask[2][2];
		}
	}
		
	imshow(str,imgClone);

	for(int i=0;i<imgClone.rows;i++)
	{
		for(int j=0;j<imgClone.cols;j++)
		{
			imgClone.at<uchar>(i,j) = img.at<uchar>(i,j)-imgClone.at<uchar>(i,j);
		}
	}

	for(int i=0;i<imgClone.rows;i++)
	{
		for(int j=0;j<imgClone.cols;j++)
		{
			imgClone.at<uchar>(i,j) = img.at<uchar>(i,j)+imgClone.at<uchar>(i,j);
		}
	}

	imshow(str1,imgClone);
	
	
}

void LoG7(Mat img, Mat imgClone, string const &str)
{
	int mask1[7][7];
	double sigma1 = 1.4;
	int min1 = -3,max1 = 3;
	
	
	double sigmaTo4 = pow(sigma1,4);
	double sigmaTo2 = pow(sigma1,2);

	double pi = 3.14;

	int x = 0;

	// for 7 x 7 matrix

	for(int i=min1;i<=max1;i++)
	{
		int y = 0;

		for(int j=min1;j<=max1;j++)
		{
			double log_value = (double)(-1.0/(pi*sigmaTo4)) * (1.0 - (i*i+j*j)/(2.0*sigmaTo2)) * (double)pow(2.7, ((-(i*i+j*j)/(2.0*sigmaTo2))));
			mask1[x][y] = int(log_value*483);
			++y;
		}
		++x;

	}
	cout<<endl<<"7 x 7"<<endl;
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			cout << mask1[i][j] << "\t";
		}
		cout << endl;
	}

	
	for(int i=0;i<7;i++)
	{
		for(int j=0;j<7;j++)
		{
			imgClone.at<uchar>(i,j)=0.0;
		}
	}


	for (int x = max1; x < img.rows - max1; x++)
	{
		for (int y = max1; y < img.cols - max1; y++)
		{
			int m = 0, q = 0;
			int value = 0;
			for (int s = min1; s <= max1; s++)
			{
				for (int t = min1; t <= max1; t++)
				{
					
					value = value + mask1[m][q] * img.at<uchar>(x + s, y + t);
					q++;
				}
				m++;
			}
			if (value < 0)
			{
				value = 0;
			}
			else
			{
				imgClone.at<uchar>(x, y) = int(value);
			}
		}
	}


imshow(str,imgClone);

}

void LoG11(Mat img, Mat imgClone,string const &str)
{
	int mask2[11][11];
	double sigma2=5.0;
	int min2 = -5,max2 = 5;
	
	

	double pi = 3.14;

	

	// 11 x 11 Mask

	double sigmaTo4 = pow(sigma2,4);
	double sigmaTo2 = pow(sigma2,2);

	int x = 0;
	for(int i=min2;i<=max2;i++)
	{
		int y = 0;

		for(int j=min2;j<=max2;j++)
		{
			double log_value = (double)(-1.0/(pi*sigmaTo4)) * (1.0 - (i*i+j*j)/(2.0*sigmaTo2)) * (double)pow(2.7, ((-(i*i+j*j)/(2.0*sigmaTo2))));
			mask2[x][y] = int(log_value*40000);
			++y;
		}
		++x;

	}
	cout<<endl<<"11 x 11"<<endl;
	for (int i = 0; i < 11; i++)
	{
		for (int j = 0; j < 11; j++)
		{
			cout << mask2[i][j] << "\t";
		}
		cout << endl;
	}


	for(int i=0;i<11;i++)
	{
		for(int j=0;j<11;j++)
		{
			imgClone.at<uchar>(i,j)=0.0;
		}
	}


	for (int x = max2; x < img.rows - max2; x++)
	{
		for (int y = max2; y < img.cols - max2; y++)
		{
			int m = 0, q = 0;
			int value = 0;
			for (int s = min2; s <= max2; s++)
			{
				for (int t = min2; t <= max2; t++)
				{
					
					value = value + mask2[m][q] * img.at<uchar>(x + s, y + t);
					q++;
				}
				m++;
			}
			if (value < 0)
			{
				value = 0;
			}
			else
			{
				imgClone.at<uchar>(x, y) = int(value);
			}
		}
	}

	
imshow(str,imgClone);

}


int main(int argc, char** argv)
{
	
	
	Mat img = imread("ant_gray.bmp", IMREAD_GRAYSCALE);
	Mat imgClone = imread("ant_gray.bmp", IMREAD_GRAYSCALE);


	Mat img1 = imread("basel_gray.bmp",IMREAD_GRAYSCALE);
	Mat img1Clone = imread("basel_gray.bmp",IMREAD_GRAYSCALE);

	// imshow("image 1",img);
	// unsharpMask(img,imgClone,"blur_1","enhance_1");
	// sobel(img,imgClone,"sobel_1");
	// LoG7(img,imgClone,"LoG7_1");
	// LoG11(img,imgClone,"LoG11_1");

	imshow("image 2",img1);
	unsharpMask(img1,img1Clone,"blur_2","enhance_2");
	sobel(img1,img1Clone,"sobel_2");
	LoG7(img1,img1Clone,"LoG7_2");
	LoG11(img1,img1Clone,"LoG11_2");
	




	waitKey(0);
	return 0;

}
