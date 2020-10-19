
#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
#define pi 3.142857 

using namespace std;
using namespace cv;

void dctCalculation(double matrix[],int n) 
{ 
	

	double res[n];

	for(int i=0;i<n;i++)
	{
		double sum=0.0;
		double weight;
		if(i==0)
			weight = sqrt(0.5);
		else
			weight = 1.0;

		for(int j=0;j<n;j++)
		{
			sum += weight * matrix[j] * cos(pi*(j+0.5)*i/n);
		}

		res[i] = sum * sqrt(2.0/n);

	}


	for(int i=0;i<n;i++)
		cout<<res[i]<<" ";
	cout<<endl;
} 

Mat RGBtoHSI(Mat img)
{

	Mat hsi(img.rows,img.cols,img.type());


	float red,green,blue,hue,saturation,intensity;

	for(int i=0;i<img.rows;i++)
	{
		for(int j=0;j<img.cols;j++)
		{
			blue = img.at<Vec3b>(i,j)[0];
			green = img.at<Vec3b>(i,j)[1];
			red = img.at<Vec3b>(i,j)[2];

			intensity = (red+green+blue)/3;


			int min = std::min(red,std::min(green,blue));

			saturation = 1 - (3*min)/(red+green+blue);

			if(saturation<0.00001)
				saturation = 0;
			else if(saturation>0.99999)
				saturation = 1;


			if(saturation!=0)
			{
				hue = acos(0.5*((red-green)+(red-blue))/sqrt(((red-green)*(red-green))+((red-blue)*(green-blue))));

				if(green<=blue)
					hue = ((360*pi)/180.0) - hue;

			}

			hsi.at<Vec3b>(i,j)[0] = 0;
			hsi.at<Vec3b>(i,j)[1] = 0;
			hsi.at<Vec3b>(i,j)[2] = intensity;
		}
	}

	return hsi;


}


Mat imgToDct(Mat img)
{

	Mat imgClone = img.clone();

	int height = img.rows - (img.rows%8);
	int width = img.cols - (img.cols%8);

	for (int m = 0; m < height; m += 8)
	{
		for (int n = 0; n < width; n += 8)					
		{
			for (int i = m; i < m + 8; i++)
			{
				for (int j = n; j < n + 8; j++)
				{
					float temp = 0.0;
					for (int x = m; x < m + 8; x++)
					{
						for (int y = n; y < n + 8; y++)				
						{
							temp += (img.at<Vec3b>(x, y)[2]) *
								(cos((((2 * x) + 1)) * ((i * pi)) / (2 * 8))) *
								(cos((((2 * y) + 1)) * ((j * pi)) / (2 * 8)));			
						}
					}
					if(i==0 && j==0)
						temp = temp * 0.125;
					else if((i!=0 && j==0) || (i==0 && j!=0))
						temp = temp * 0.25 * (1/sqrt(2)); 
					else if(i!=0 && j!=0)
						temp = temp * 0.25;
					imgClone.at<Vec3b>(i, j)[2] = int(temp);				
				}

			}
		}
	}


	return imgClone;

}

Mat dcComponent(Mat img)
{
	Mat imgClone = img.clone();
	int height = img.rows - (img.rows % 8);
	int width = img.cols - (img.cols % 8);
	
	for (int m = 0; m < height; m = m + 8)
	{
		for (int n = 0; n < width; n = n + 8)					
		{
			for (int i = m; i < m + 8; i++)
			{
				for (int j = n; j < n + 8; j++)
				{
					if (i == m && j == n)
						continue;
					else					
						imgClone.at<Vec3b>(i, j)[2] = 0;
				}
			}
		}
	}
	return imgClone;
}

Mat imgToD2(Mat img)
{	

	Mat imgClone = img.clone();
	int height = img.rows - (img.rows % 8);
	int width = img.cols - (img.cols % 8);
	
	for (int m = 0; m < height; m += 8)
	{
		for (int n = 0; n < width; n += 8)
		{
			for (int i = m; i < m + 8; i++)
			{
				for (int j = n; j < n + 8; j++)
				{
					if (i == m && j == n)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else if(i == m && j == n+1)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else if (i == m && j == n + 2)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else if (i == m && j == n + 3)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else if (i == m + 1 && j == n)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else if (i == m + 1 && j == n+1)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else if (i == m + 1 && j == n+2)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else if (i == m + 2 && j == n)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else if (i == m + 2 && j == n + 1)
					{
						imgClone.at<Vec3b>(i, j)[2] = imgClone.at<Vec3b>(i, j)[2];
					}
					else
					{
						imgClone.at<Vec3b>(i, j)[2] = 0;
					}
				}
			}
		}
	}
	return imgClone;
}

Mat Idct(Mat img)
{	

	Mat imgClone = img.clone();
	int height = img.rows - (img.rows % 8);
	int width = img.cols - (img.cols % 8);
	
	for (int m = 0; m < height; m = m+8)
	{
		for (int n = 0; n < width; n = n+8)					
		{
			for (int i = m; i < m + 8; i++)
			{
				for (int j = n; j < n + 8; j++)
				{
					float temp = 0.0;
					for (int x = m; x < m + 8; x++)
					{
						for (int y = n; y < n + 8; y++)				
						{
							temp = temp + img.at<Vec3b>(x, y)[2] * cos(((2 * x) + 1) * (i * pi) / 16) * cos(((2 * y) + 1) * (j * pi) / 16);
											
							if(x==0 && y==0)
								temp = temp * 0.5;
							else if((x!=0 && y==0) || (x==0 && y!=0))
								temp = temp * (1/sqrt(2)); 
							
						}
					}
					
					imgClone.at<Vec3b>(i, j)[2] = int(temp);

				}
			}
		}
	}

	return imgClone;
}


int xGradient(Mat img,int x,int y)
{
	return (img.at<Vec3b>(x-1,y-1)[2] + 2*img.at<Vec3b>(x,y-1)[2] + img.at<Vec3b>(x+1,y-1)[2] - img.at<Vec3b>(x-1,y+1)[2] - 2*img.at<Vec3b>(x,y+1)[2] - img.at<Vec3b>(x+1,y+1)[2] );
}

int yGradient(Mat img,int x,int y)
{
	return (img.at<Vec3b>(x-1,y-1)[2] + 2*img.at<Vec3b>(x-1,y)[2] + img.at<Vec3b>(x-1,y+1)[2] - img.at<Vec3b>(x+1,y-1)[2] - 2*img.at<Vec3b>(x+1,y)[2] - img.at<Vec3b>(x+1,y+1)[2] );
}

Mat sobel(Mat hsi)
{

	int gx,gy,gfinal;

	Mat imgClone = hsi.clone();

	for(int i=0;i<hsi.rows;i++)
	{
		for(int j=0;j<hsi.cols;j++)
		{
			imgClone.at<Vec3b>(i,j)[0] = 0.0;
			imgClone.at<Vec3b>(i,j)[1] = 0.0;
			imgClone.at<Vec3b>(i,j)[2] = 0.0;
		}
	}

	
	for(int i=1;i<hsi.rows-1;i++)
	{
		for(int j=1;j<hsi.cols-1;j++)
		{	
			gx = xGradient(hsi,i,j);
			gy = yGradient(hsi,i,j);
			gfinal = abs(gx) + abs(gy);

			if(gfinal>255)
				gfinal = 255;
			if(gfinal<0)
				gfinal = 0;

			imgClone.at<Vec3b>(i,j)[2] = gfinal;
		}
	}

	return imgClone;

	

}

int main(int argc, char** argv)
{
	
double matrix1[] = {10,11,12,11,12,13,12,11};
double matrix2[] = {10,-10,8,-7,8,-8,7,-7};
double matrix3[] = {10,11,12,11,12,13,12,11,10,-10,8,-7,8,-8,7,-7};

				 
dctCalculation(matrix1,8); 
dctCalculation(matrix2,8);
dctCalculation(matrix3,16);


	Mat img = imread("basel3.bmp", IMREAD_COLOR);
	Mat img1 = imread("Building1.bmp",IMREAD_COLOR);
	Mat img2 = imread("Disk.bmp",IMREAD_COLOR);
	
	Mat I,I1,I2,F,D1,D2,R1,R2,EdgeImage1,EdgeImage2;

	imshow("Original Image",img);

	I = RGBtoHSI(img);

	imshow("Intensity Image",I);

	F = imgToDct(I);

	imshow("Frequency Image",F);

	D1 = dcComponent(F);

	imshow("D1 Image",D1);

	D2 = imgToD2(F);

	imshow("D2 Image",D2);

	R1 = Idct(D1);

	imshow("R1 Image",R1);

	R2 = Idct(D2);

	imshow("R2 Image",R2);

	I1 = RGBtoHSI(img1);
	I2 = RGBtoHSI(img2);

	EdgeImage1 = sobel(I1);
	EdgeImage2 = sobel(I2);

	imshow("Sobel1",EdgeImage1);
	imshow("Sobel2",EdgeImage2);


		waitKey(0);

		return 0;

}