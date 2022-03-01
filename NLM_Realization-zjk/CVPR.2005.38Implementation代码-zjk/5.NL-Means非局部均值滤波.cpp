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
//����Ϊ�˲�����ĺ���

uchar** NLmeansFiltering(uchar** pic, int row, int col, int ssize, int ksize, double h);//��pic���зǾֲ���ֵ�˲���ssizeΪ��������С,ksizeΪ������С��hΪmse��������


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
			dif[i][j] = 5*x;
		}
	}
	showAndstore(dif, row, col);
}

int main() {
	uchar** gra;//ͼƬ�Ķ�ά����
	int h, w;//��Ƭ�����ظ�����
	gra = convertVec("..\\Laughing.png", h, w);//Ԥ����ͼƬת��Ϊ����

	uchar** res = NULL;//�洢�˲��������
	int ssize,ksize;//ssizeΪ��������С,ksizeΪ������С
	double hc;//����������hԽ��ȥ��Խǿ��Խģ��
	printf("Enter the odd search size (int && odd): "); cin >> ssize;
	printf("Enter the odd block size (int && odd): "); cin >> ksize;
	printf("Enter the Mse constant (double): "); cin >> hc;
	res = NLmeansFiltering(gra, h, w, ssize, ksize, hc);
	showAndstore(res, h, w);

	//��ʾ����
	//showDifference(gra, res, h, w);
	return 0;
}
uchar** NLmeansFiltering(uchar** pic, int row, int col, int ssize, int ksize, double h) {
	//��pic���зǾֲ���ֵ�˲���ssizeΪ��������С > ksizeΪ������С��hΪmse��������
	int kh = ksize / 2, sh = ssize / 2;
	int ad = kh + sh;
	double** tep = new double* [row + 2*ad];//������Ϊ�˱������ȣ�ȫ��ȡΪ���������м���
	for (int i = 0; i < row+ 2 * ad; i++) {
		//ת�������������ͣ������������������ksize+ssize��һ�룩�����þ������
		tep[i] = new double[col + 2 * ad];
		for (int j = 0; j < col+ 2 * ad; j++) {
			//���Ÿ���������
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
		//����ͼ��
		for (int j = ad; j < ad + col; j++) {
			//����i��j����ֵ
			double sum=0;//��һ��ϵ��
			double w;//δ��һ����Ȩ
			double ans=0;//�õ��ֵ
			for (int si = i - sh; si <= i + sh; si++) {
				for (int sj = j - sh; sj <= j + sh; sj++) {
					//������������B:si,sj���A:i,j���mse
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
	//ȡtepΪ��չ���򷵻�
	uchar** res = new uchar * [row];
	for (int i = 0; i < row; i++) {
		res[i] = new uchar[col];
		for (int j = 0; j < col; j++) {
			res[i][j] = (uchar)temp[i + ad][j + ad];
		}
	}
	return res;
}