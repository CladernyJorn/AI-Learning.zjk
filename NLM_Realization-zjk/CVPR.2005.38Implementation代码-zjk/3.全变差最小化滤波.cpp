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
//以下为各向异性滤波处理的函数
uchar** TotalVariationFiltering(uchar** pic, int row, int col, int iteration, double ep,double lambda, double dt);//对pic进行全变差最小化滤波,iteration为总迭代次数，后两个系数为迭代中的平滑系数和学习步长Δt


void showpic(string c) {
	//显示c地址的图片
	cv::Mat img = cv::imread(c, 0);
	cv::imshow("test", img);
	cv::waitKey(0);
	system("pause");
}
uchar** convertVec(string c, int& height, int& width) {
	//将地址c处的图片转到二维数组gra，用后两个参数记录长宽
	cv::Mat img = cv::imread(c, 0);
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
void showAndstore(uchar** array, int height, int width) {
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

void showDifference(uchar** pic, uchar** res, int row, int col) {
	//将处理前后两张图片做差取绝对值并以图片展示导出
	uchar** dif = new uchar * [row];
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
	gra = convertVec("..\\lena1.png", h, w);//预处理图片转化为数组
	uchar** res = NULL;//存储滤波后的数组
	double ep,lambda, dt;
	int iternum;
	printf("Enter the three parameters of TVFiltering:\n");
	printf("iteration  (int 1-50)= "); cin >> iternum;
	printf("gradient regularization  (double around 1)= "); cin >> ep;
	printf("λ (double 0-0.05)= "); cin >> lambda;
	printf("Δt (double 0-0.5)= "); cin >> dt;
	res = TotalVariationFiltering(gra, h, w, iternum, ep,lambda, dt);
	showAndstore(res, h, w);

	//显示区别
	//showDifference(gra, res, h, w);
	return 0;
}
uchar** TotalVariationFiltering(uchar** pic, int row, int col, int iteration, double ep,double lambda, double dt) {
	//对pic进行总变分最小化滤波
	////对pic进行全变差最小化滤波,iteration为总迭代次数，后两个系数为迭代中的平滑系数和学习步长Δt
	double** tep = new double* [row];//迭代中为了保留精度，全部取为浮点数进行计算
	double** temp = new double* [row];
	for (int i = 0; i < row; i++) {
		//转换数组数据类型
		tep[i] = new double[col];
		temp[i] = new double[col];
		for (int j = 0; j < col; j++) {
			tep[i][j] = (double)pic[i][j];
			temp[i][j] = (double)pic[i][j];
		}
	}

	for (int cnt = 0; cnt < iteration; cnt++) {
		//iteration次迭代,边界的一层像素点不处理
		for (int i = 1; i < row - 1; i++) {
			for (int j = 1; j < col - 1; j++) {
				double ux, uy, uxx, uyy, uxy;
				ux = (temp[i + 1][j] - temp[i - 1][j]) / 2.0;
				uy = (temp[i][j + 1] - temp[i][j - 1]) / 2.0;
				uxx = temp[i + 1][j] + temp[i - 1][j] - 2 * temp[i][j];
				uyy = temp[i][j + 1] + temp[i][j - 1] - 2 * temp[i][j];
				uxy = (temp[i + 1][j + 1] + temp[i - 1][j - 1] - temp[i - 1][j + 1] - temp[i + 1][j - 1])/4.0;
				tep[i][j] += dt * (lambda * ((double)pic[i][j] - temp[i][j]) + ((ux * ux + ep * ep) * uyy + (uy * uy + ep * ep) * uxx - 2 * ux * uy * uxy) / pow(ux * ux + uy * uy + ep * ep, 1.5));
			}
		}
		for (int i = 1; i < row - 1; i++)
			for (int j = 1; j < col - 1; j++)
				temp[i][j] = tep[i][j];
	}
	uchar** res = new uchar * [row];
	for (int i = 0; i < row; i++) {
		//转换回uchar数组数据类型到res
		res[i] = new uchar[col];
		for (int j = 0; j < col; j++)
			res[i][j] = (uchar)tep[i][j];
	}
	return res;
}
