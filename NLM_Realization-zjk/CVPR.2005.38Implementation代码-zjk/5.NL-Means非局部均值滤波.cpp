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
//以下为滤波处理的函数

uchar** NLmeansFiltering(uchar** pic, int row, int col, int ssize, int ksize, double h);//对pic进行非局部均值滤波，ssize为搜索窗大小,ksize为邻域块大小，h为mse评估参数


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
	gra = convertVec("..\\Laughing.png", h, w);//预处理图片转化为数组

	uchar** res = NULL;//存储滤波后的数组
	int ssize,ksize;//ssize为搜索窗大小,ksize为邻域块大小
	double hc;//评估参数（h越大，去噪越强，越模糊
	printf("Enter the odd search size (int && odd): "); cin >> ssize;
	printf("Enter the odd block size (int && odd): "); cin >> ksize;
	printf("Enter the Mse constant (double): "); cin >> hc;
	res = NLmeansFiltering(gra, h, w, ssize, ksize, hc);
	showAndstore(res, h, w);

	//显示区别
	//showDifference(gra, res, h, w);
	return 0;
}
uchar** NLmeansFiltering(uchar** pic, int row, int col, int ssize, int ksize, double h) {
	//对pic进行非局部均值滤波，ssize为搜索窗大小 > ksize为邻域块大小，h为mse评估参数
	int kh = ksize / 2, sh = ssize / 2;
	int ad = kh + sh;
	double** tep = new double* [row + 2*ad];//迭代中为了保留精度，全部取为浮点数进行计算
	for (int i = 0; i < row+ 2 * ad; i++) {
		//转换数组数据类型，并将长宽扩大（两侧各ksize+ssize的一半），并用镜面填充
		tep[i] = new double[col + 2 * ad];
		for (int j = 0; j < col+ 2 * ad; j++) {
			//共九个区域镜像处理
			if (i <= ad) {
				if (j <= ad)tep[i][j] = (double)pic[2 * ad - i-ad][2 * ad - j-ad];
				else if (j <= ad + col - 1)tep[i][j] = (double)pic[2 * ad - i-ad][j-ad];
				else tep[i][j] = (double)pic[2 * ad - i-ad][2 * (ad + col - 1) - j-ad];
			}
			else if (i >= ad + row - 1) {
				if (j <= ad)tep[i][j] = (double)pic[2 * (ad + row - 1) - i-ad][2 * ad - j-ad];
				else if (j <= ad + col - 1)tep[i][j] = (double)pic[2 * (ad + row - 1) - i-ad][j-ad];
				else tep[i][j] = (double)pic[2 * (ad + row - 1) - i-ad][2 * (ad + col - 1) - j-ad];
			}
			else {
				if (j <= ad)tep[i][j] = (double)pic[i - ad][2 * ad - j - ad];
				else if (j >= ad + col - 1)tep[i][j] = (double)pic[i - ad][2 * (ad + col - 1) - j - ad];
				else tep[i][j] = (double)pic[i - ad][j - ad];
			}
		}
	}
	double** temp = new double* [row + 2 * ad];
	for (int i = ad; i < ad + row; i++) {
		temp[i] = new double[col + 2 * ad];
		//处理图像
		for (int j = ad; j < ad + col; j++) {
			//计算i，j出的值
			double sum=0;//归一化系数
			double w;//未归一化的权
			double ans=0;//该点的值
			for (int si = i - sh; si <= i + sh; si++) {
				for (int sj = j - sh; sj <= j + sh; sj++) {
					//计算搜索框内B:si,sj点对A:i,j点的mse
					double mse = 0;
					for (int m = -kh; m <= kh; m++) {
						for (int n = -kh; n <= kh; n++) {
							mse += (tep[i + m][j + n] - tep[si + m][sj + n]) * (tep[i + m][j + n] - tep[si + m][sj + n]);
						}
					}
					mse /= (double)ksize * ksize;
					w = exp(-mse / (h * h));
					sum += w;
					ans += w * tep[si][sj];
				} 
			}
			ans /= sum;
			temp[i][j] = ans;
		}
	}
	//取tep为扩展区域返回
	uchar** res = new uchar * [row];
	for (int i = 0; i < row; i++) {
		res[i] = new uchar[col];
		for (int j = 0; j < col; j++) {
			res[i][j] = (uchar)temp[i + ad][j + ad];
		}
	}
	return res;
}