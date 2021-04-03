#pragma once
#include<cmath>
#include"initKey.cuh"
void getSine2DSequence(double* X, double* Y, int N);//获取2D HSM混沌序列.序列长度为N
void getLogisticSequence(double* X, int n);//获取logistic混沌序列，n为序列长度