#pragma once
#include "device_launch_parameters.h"
#include"dna.cuh"
#include"initKey.cuh"

//获取dna加密图像所需的序列
__global__ void getDnaProSequenceKernel(unsigned char* dev_dnaProSequence, double* update, int* dev_chaos, int rows,int cols);
//对图像进行dna编码
__global__ void  imgDnaEncodeKernel(unsigned char* dev_img, unsigned char* dev_dna, unsigned char* dev_dnaProSequence, int rows, int cols);
//cuda测试
__global__ void test(unsigned char* dev_dna, int n);
__global__ void test(int* seq, int n);

//获取dna异或序列
__global__ void getDnaXorSequenceKernel(unsigned char* dev_dnaXor, unsigned char* dev_dnaProSequence, int rows, int cols);


//获取dna加序列
__global__ void getDnaAddSequenceKernel(unsigned char* dev_dnaAdd, unsigned char* dev_dnaProSequence, int rows, int cols);

//对dna序列执行异或操作
__global__ void dnaXorKernel(unsigned char* dev_dna, unsigned char* dev_dnaXor, int size);
//对dna序列执行加操作
__global__ void dnaAddKernel(unsigned char* dev_dna, unsigned char* dev_dnaXor, int size);

//对dna序列解码
__global__ void dnaDecodeKernel(unsigned char* dev_dna, unsigned char* img, unsigned char* dev_dnaProSequence, int rows,int cols);


//下面是解密函数
//对图像进行dna编码
__global__ void  de_imgDnaEncodeKernel(unsigned char* dev_img, unsigned char* dev_dna, unsigned char* dev_dnaProSequence, int rows, int cols);
//对dna序列执行减操作
__global__ void dnaMinusKernel(unsigned char* dev_dna, unsigned char* dev_dnaXor, int cols ,int size);
//对dna序列解码
__global__ void de_dnaDecodeKernel(unsigned char* dev_dna, unsigned char* img, unsigned char* dev_dnaProSequence, int rows,int cols);

//DNA-S盒替换
__global__ void dna_sbox_replaceKernel(unsigned char* dev_dna, unsigned char* dev_dnaProSequence, int rows, int cols);
//DNA-S盒替换逆运算
__global__ void de_dna_sbox_replaceKernel(unsigned char* dev_dna, unsigned char* dev_dnaProSequence, int rows, int cols);