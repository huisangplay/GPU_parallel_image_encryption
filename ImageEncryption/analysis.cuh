#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include<fstream>
using namespace cv;
void imgWriteTofileByBinary(Mat img);//将图像每个像素的值以二进制方式写入dat文件
double getChiSquareTests(Mat img);