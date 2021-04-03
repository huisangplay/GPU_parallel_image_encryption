#pragma once
#include "cuda_runtime.h"
__device__ void dnaEncode(unsigned char c, unsigned char* dna, unsigned char n);
__device__ unsigned char dnaXor(unsigned char a, unsigned char b);
__device__ unsigned char dnaAdd(unsigned char a, unsigned char b);
__device__ unsigned char dnaDecode(unsigned char* dna, unsigned char n);
__device__ unsigned char dnaMinus(unsigned char a, unsigned char b);
__device__ int dna_to_num(unsigned char* dna,int dna_rule);
__device__ char num_to_dna(int num, int dna_rule);