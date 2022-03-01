#include<cstdio>
#include<iostream>
#include<cmath>
#include<string>
#include<cstdlib>
#include<cctype>
#include <opencv2\opencv.hpp>
#define pi 3.1415926535898
using namespace std;



void showpic(string c);//显示c地址的图片
uchar** convertVec(string c, int& height, int& width);//将地址c处的图片转到二维数组gra，用后两个参数记录长宽
void showAndstore(uchar** array, int height, int width);//将array数组的导出图片显示并到文件夹
void showDifference(uchar** pic, uchar** res);//将处理前后两张图片做差取绝对值并以图片展示导出
//以下为高斯滤波处理的函数

uchar** gauseFiltering(uchar** pic,int row, int col, int size, double sx);//对pic进行高斯滤波,size为卷积核边长(默认奇数）,sx为单变量标准差sigma
double** kernalOfgause(int size, double sx);//生成高斯滤波卷积核，参数定义同上


void showpic(string c) {
	//显示c地址的图片
	cv::Mat img = cv::imread(c,0);
	cv::imshow("test", img);
	cv::waitKey(0);
	system("pause");
}
uchar** convertVec(string c, int& height, int& width) {
	//将地址c处的图片转到二维数组gra，用后两个参数记录长宽
	cv::Mat img = cv::imread(c,0);
	uchar** array = new uchar * [img.rows];
	width = img.cols;
	height = img.rows;
	for (int i = 0; i < height; i++)
		array[i] = new uchar[width];
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			array[i][j] = img.at<uchar>(i, j);
		}
	}
	return array;
}
void showAndstore(uchar** array, int height, int width){
	//将array数组的导出图片显示并到文件夹
	cv::Mat img(height, width, CV_8UC1);
	uchar* ptmp = NULL;
	for (int i = 0; i < height; ++i)
	{
		ptmp = img.ptr<uchar>(i);

		for (int j = 0; j < width; ++j)
		{
			ptmp[j] = array[i][j];
		}
	}
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);    // 无压缩png.
	compression_params.push_back(cv::IMWRITE_PNG_STRATEGY);
	compression_params.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);
	cv::imwrite("..\\result.png", img, compression_params);
	showpic("..\\result.png");
}

void showDifference(uchar** pic, uchar** res,int row,int col) {
	//将处理前后两张图片做差取绝对值并以图片展示导出
	uchar** dif = new uchar*[row];
	for (int i = 0; i < row; i++) {
		dif[i] = new uchar[col];
		for (int j = 0; j < col; j++) {
			int x = res[i][j] - pic[i][j];
			x = x > 0 ? x : -x;
			dif[i][j] = 5*x;
		}
	}
	showAndstore(dif, row, col);
}

int main() {
	uchar** gra;//图片的二维数组
	int h, w;//照片宽（像素个数）
	gra = convertVec("..\\Laughing.png", h, w);//预处理图片转化为数组
	
	uchar** res=NULL;//存储滤波后的数组
	int size;//卷积核边长
	double sx;//卷积核标准差
	printf("Enter the odd size of kernal (int && odd): "); cin >> size;
	printf("Enter the standard deviation of kernal (double): "); cin >> sx;
	res=gauseFiltering(gra,h,w,size,sx);
	showAndstore(res, h, w);

	//显示区别
	//showDifference(gra, res, h, w);
	return 0;
}
uchar** gauseFiltering(uchar** pic, int row, int col, int size, double sx){
	//对pic进行高斯滤波到des,size为卷积核边长,sx为单变量标准差sigma
	double** core=NULL;
	core=kernalOfgause(size, sx);
	uchar** res = new uchar * [row];
	//滤波处理
	for (int i = 0; i < row; i++) {
		res[i] = new uchar[col];
		for (int j = 0; j < col; j++) {
			//对每个像素点
			double ans = 0;
			int c = size / 2;
			int i0 = i - c, j0 = j - c;
			for (int m = 0; m < size; m++) {
				for (int n = 0; n < size; n++) {
					//边界部分采用镜面对称方法
					int p, q;//镜像后的像素位置
					if (i0 + m > row - 1)p = 2 * (row - 1) - (i0 + m);
					else if (i0 + m < 0)p = 0 - (i0 + m);
					else p = i0 + m;
					if (j0 + n > col - 1)q = 2 * (col - 1) - (j0 + n);
					else if (j0 + n < 0)q = 0 - (j0 + n);
					else q = j0 + n;
					ans += (double)pic[p][q] * core[m][n];
				}
			}
			res[i][j] = (uchar)ans;
		}
	}
	return res;
}
double** kernalOfgause(int size, double sx) {
	//生成高斯滤波卷积核
	double** ker = new double* [size];
	int c = size / 2;
	double sum = 0;
	for (int i = 0; i < size; i++) {
		ker[i] = new double[size];//开辟size*size卷积核空间
		for (int j = 0; j < size; j++) {
			ker[i][j] = exp(-((double)((i - c) * (i - c) + (j - c) * (j - c))) / (2 * sx * sx));
			sum += ker[i][j];
		}
	}
	//归一化
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			ker[i][j] /= sum;
	}
	return ker;
}