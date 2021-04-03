#pragma once
#include "cuda_runtime.h"
//sine混沌映射初始值和参数值
extern double std_a;//0-6*pi
extern double std_b ;//0-6*pi
extern double std_k;//>0
extern double pi;//
//三维猫映射初始值和参数值
extern double cat3D_x;//0-1
extern double cat3D_y ;//0-1
extern double cat3D_z ;//0-1
//洛伦兹混沌映射初始值和参数值
extern double lorenz_y;//0-1
extern double lorenz_z;//0-1
extern double lorenz_q ;//0-1
extern __device__ int dev_lorenz_f ;
extern __device__ int dev_lorenz_r;
extern __device__ double dev_lorenz_g;

//sine2D混沌映射初始值和参数值
extern double sine2D_a;//范围--负无穷到正无穷
extern double sine2D_b ;//范围--负无穷到正无穷

extern double sine2D_x ;//0-1
extern double sine2D_y ;//0-1

//logistic混沌映射初始值和参数值
extern double logistic_u ;//参数u的范围(3.5699456, 4]
extern double logistic_x ;//0-1

/*
一共14个需要更新的值，前十二个利用前24组哈希值
剩下两个利用剩下的八个哈希值
*/
extern __device__ double dev_std_a;//0-1
extern __device__ double dev_std_b;//0.87-1
extern __device__ double dev_std_k;//0.87-1
//三维猫映射初始值和参数值
extern __device__ double dev_cat3D_x;//0-1
extern __device__ double dev_cat3D_y;//0-1
extern __device__ double dev_cat3D_z;//0-1
//洛伦兹混沌映射初始值和参数值
extern __device__ double dev_lorenz_y;//0-1
extern __device__ double dev_lorenz_z;//0-1
extern __device__ double dev_lorenz_q;//0-1
extern __device__ double dev_pi;//0-1

extern __device__ char dna_sbox[1024];