#include<cstdio>
#include<iostream>
#include<cmath>
#include<string>
#include<cstdlib>
#include<cctype>
#include <opencv2\opencv.hpp>
#define pi 3.1415926535898
using namespace std;



void showpic(string c);//��ʾc��ַ��ͼƬ
uchar** convertVec(string c, int& height, int& width);//����ַc����ͼƬת����ά����gra���ú�����������¼����
void showAndstore(uchar** array, int height, int width);//��array����ĵ���ͼƬ��ʾ�����ļ���
void showDifference(uchar** pic, uchar** res);//������ǰ������ͼƬ����ȡ����ֵ����ͼƬչʾ����
//����Ϊ��˹�˲�����ĺ���

uchar** gauseFiltering(uchar** pic,int row, int col, int size, double sx);//��pic���и�˹�˲�,sizeΪ����˱߳�(Ĭ��������,sxΪ��������׼��sigma
double** kernalOfgause(int size, double sx);//���ɸ�˹�˲�����ˣ���������ͬ��


void showpic(string c) {
	//��ʾc��ַ��ͼƬ
	cv::Mat img = cv::imread(c,0);
	cv::imshow("test", img);
	cv::waitKey(0);
	system("pause");
}
uchar** convertVec(string c, int& height, int& width) {
	//����ַc����ͼƬת����ά����gra���ú�����������¼����
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
	//��array����ĵ���ͼƬ��ʾ�����ļ���
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
	compression_params.push_back(0);    // ��ѹ��png.
	compression_params.push_back(cv::IMWRITE_PNG_STRATEGY);
	compression_params.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);
	cv::imwrite("..\\result.png", img, compression_params);
	showpic("..\\result.png");
}

void showDifference(uchar** pic, uchar** res,int row,int col) {
	//������ǰ������ͼƬ����ȡ����ֵ����ͼƬչʾ����
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
	uchar** gra;//ͼƬ�Ķ�ά����
	int h, w;//��Ƭ�����ظ�����
	gra = convertVec("..\\Laughing.png", h, w);//Ԥ����ͼƬת��Ϊ����
	
	uchar** res=NULL;//�洢�˲��������
	int size;//����˱߳�
	double sx;//����˱�׼��
	printf("Enter the odd size of kernal (int && odd): "); cin >> size;
	printf("Enter the standard deviation of kernal (double): "); cin >> sx;
	res=gauseFiltering(gra,h,w,size,sx);
	showAndstore(res, h, w);

	//��ʾ����
	//showDifference(gra, res, h, w);
	return 0;
}
uchar** gauseFiltering(uchar** pic, int row, int col, int size, double sx){
	//��pic���и�˹�˲���des,sizeΪ����˱߳�,sxΪ��������׼��sigma
	double** core=NULL;
	core=kernalOfgause(size, sx);
	uchar** res = new uchar * [row];
	//�˲�����
	for (int i = 0; i < row; i++) {
		res[i] = new uchar[col];
		for (int j = 0; j < col; j++) {
			//��ÿ�����ص�
			double ans = 0;
			int c = size / 2;
			int i0 = i - c, j0 = j - c;
			for (int m = 0; m < size; m++) {
				for (int n = 0; n < size; n++) {
					//�߽粿�ֲ��þ���ԳƷ���
					int p, q;//����������λ��
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
	//���ɸ�˹�˲������
	double** ker = new double* [size];
	int c = size / 2;
	double sum = 0;
	for (int i = 0; i < size; i++) {
		ker[i] = new double[size];//����size*size����˿ռ�
		for (int j = 0; j < size; j++) {
			ker[i][j] = exp(-((double)((i - c) * (i - c) + (j - c) * (j - c))) / (2 * sx * sx));
			sum += ker[i][j];
		}
	}
	//��һ��
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			ker[i][j] /= sum;
	}
	return ker;
}