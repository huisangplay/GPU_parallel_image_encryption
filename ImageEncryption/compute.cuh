#pragma once
#include<opencv2/highgui/highgui.hpp>
#include"chaos.cuh"
#include"sha256.h"
using namespace cv;
void imgRowExchange(Mat& img, int* p);//交换图像的两行
void imgColExchange(Mat& img, int* q);//交换图像的两列
void imgConfusion(Mat& img);//将图像执行混乱操作
void chaosSelectByCol(int* chaos, double* update, int n);//通过列选择所使用的混沌系统
void updateKeys(unsigned char buf[]);//根据图像的sha256值更新密钥

//下面为图像解密函数
void deImgConfusion(Mat &img);//图像的逆混乱操作
void deImgRowExchange(Mat& img, int* p);//交换图像的两行的逆操作
void deImgColExchange(Mat& img, int* q);//交换图像的两列的逆操作