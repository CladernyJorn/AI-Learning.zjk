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
//����Ϊ���������˲�����ĺ���
uchar** anisotropicfiltering(uchar** pic, int row, int col, int iteration,double lambda,double k);//��pic���и��������˲�������PM����,iterationΪ�ܵ���������������ϵ��Ϊ�����е�ƽ��ϵ����g�������ݶ�ģ��ֵ�������뱨���ĵ��е�������ȫһ��


void showpic(string c) {
	//��ʾc��ַ��ͼƬ
	cv::Mat img = cv::imread(c, 0);
	cv::imshow("test", img);
	cv::waitKey(0);
	system("pause");
}
uchar** convertVec(string c, int& height, int& width) {
	//����ַc����ͼƬת����ά����gra���ú�����������¼����
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

void showDifference(uchar** pic, uchar** res, int row, int col) {
	//������ǰ������ͼƬ����ȡ����ֵ����ͼƬչʾ����
	uchar** dif = new uchar * [row];
	for (int i = 0; i < row; i++) {
		dif[i] = new uchar[col];
		for (int j = 0; j < col; j++) {
			int x = res[i][j] - pic[i][j];
			x = x > 0 ? x : -x;
			dif[i][j] =5*x;
		}
	}
	showAndstore(dif, row, col);
}

int main() {
	uchar** gra;//ͼƬ�Ķ�ά����
	int h, w;//��Ƭ�����ظ�����
	gra = convertVec("..\\lena1.png", h, w);//Ԥ����ͼƬת��Ϊ����
	uchar** res = NULL;//�洢�˲��������
	double lambda, k;
	int iternum;
	printf("Enter the three parameters:\n");
	printf("iteration  (int 1-50)= "); cin >> iternum;
	printf("�� (double 0-0.2)= "); cin >> lambda;
	printf("k (double 5-100)= "); cin >> k;
	res = anisotropicfiltering(gra, h, w, iternum, lambda, k);
	showAndstore(res, h, w);

	//��ʾ����
	//showDifference(gra, res, h, w);
	return 0;
}
uchar** anisotropicfiltering(uchar** pic, int row, int col, int iteration, double lambda, double k) {
	//��pic���и��������˲�������PM����,ϵ������ȡָ������
	//iterationΪ�ܵ���������������ϵ��Ϊ�����е�ƽ��ϵ����g�������ݶ�ģ��ֵ�������뱨���ĵ��е�������ȫһ��
	double** tep = new double* [row];//������Ϊ�˱������ȣ�ȫ��ȡΪ���������м���
	double** temp = new double* [row];//�����ݴ���һ�εĵ������
	for (int i = 0; i < row; i++) {
		//ת��������������
		tep[i] = new double[col];
		temp[i] = new double[col];
		for (int j = 0; j < col; j++) {
			tep[i][j] = (double)pic[i][j];
			temp[i][j] = (double)pic[i][j];
		}
	}

	for (int cnt = 0; cnt < iteration; cnt++) {
		//iteration�ε���,�߽��һ�����ص㲻����
		for (int i = 1; i < row-1; i++) {
			for (int j = 1; j < col-1; j++) {
				double ce, cw, cs, cn;
				double de, dw, ds, dn;//�������������
				//����4�������ݶ�
				de = temp[i][j + 1] - temp[i][j];
				dw = temp[i][j - 1] - temp[i][j];
				ds = temp[i + 1][j] - temp[i][j];
				dn = temp[i - 1][j] - temp[i][j];
				ce = exp(-(de / k) * (de / k));
				cw = exp(-(dw / k) * (dw / k));
				cs = exp(-(ds / k) * (ds / k));
				cn = exp(-(dn / k) * (dn / k));
				tep[i][j] += lambda * (ce * de + cw * dw + cs * ds + cn * dn);
			}
		}
		//��tep������temp
		for (int i = 1; i < row - 1; i++) 
			for (int j = 1; j < col - 1; j++) 
				temp[i][j] = tep[i][j];
	}
	uchar** res = new uchar * [row];
	for (int i = 0; i < row; i++) {
		//ת����uchar�����������͵�res
		res[i] = new uchar[col];
		for (int j = 0; j < col; j++)
			res[i][j] = (uchar)tep[i][j];
	}
	return res;
}
 