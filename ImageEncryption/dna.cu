#include "dna.cuh"

__device__ unsigned char dnaXor(unsigned char a, unsigned char b)
{
    unsigned char result = 0;
    if (a == 'A')	result = b;
    else if (b == 'A')	result = a;
    else if (a == b)	result = 'A';
    else if (a == 'G' && b == 'C' || a == 'C' && b == 'G')	result = 'T';
    else if (a == 'G' && b == 'T' || a == 'T' && b == 'G')	result = 'C';
    else if (a == 'C' && b == 'T' || a == 'T' && b == 'C')	result = 'G';
    return result;
}
__device__ void dnaEncode(unsigned char c, unsigned char* dna, unsigned char n)
{
    int a[4];
    a[0] = (c & 192) / 64;
    a[1] = (c & 48) / 16;
    a[2] = (c & 12) / 4;
    a[3] = c & 3;

    if (n == 0) {
        for (int i = 0; i < 4; i++) {
            switch (a[i]) {
            case 0:dna[i] = 'A'; break;
            case 1:dna[i] = 'G'; break;
            case 2:dna[i] = 'C'; break;
            case 3:dna[i] = 'T'; break;
            }
        }
    }
    else if (n == 1) {
        for (int i = 0; i < 4; i++) {
            switch (a[i]) {
            case 0:dna[i] = 'A'; break;
            case 1:dna[i] = 'C'; break;
            case 2:dna[i] = 'G'; break;
            case 3:dna[i] = 'T'; break;
            }
        }
    }
    else if (n == 2) {
        for (int i = 0; i < 4; i++) {
            switch (a[i]) {
            case 0:dna[i] = 'G'; break;
            case 1:dna[i] = 'A'; break;
            case 2:dna[i] = 'T'; break;
            case 3:dna[i] = 'C'; break;
            }
        }
    }
    else if (n == 3) {
        for (int i = 0; i < 4; i++) {
            switch (a[i]) {
            case 0:dna[i] = 'C'; break;
            case 1:dna[i] = 'A'; break;
            case 2:dna[i] = 'T'; break;
            case 3:dna[i] = 'G'; break;
            }
        }
    }
    else if (n == 4) {
        for (int i = 0; i < 4; i++) {
            switch (a[i]) {
            case 0:dna[i] = 'C'; break;
            case 1:dna[i] = 'T'; break;
            case 2:dna[i] = 'A'; break;
            case 3:dna[i] = 'G'; break;
            }
        }
    }
    else if (n == 5) {
        for (int i = 0; i < 4; i++) {
            switch (a[i]) {
            case 0:dna[i] = 'G'; break;
            case 1:dna[i] = 'T'; break;
            case 2:dna[i] = 'A'; break;
            case 3:dna[i] = 'C'; break;
            }
        }
    }
    else if (n == 6) {
        for (int i = 0; i < 4; i++) {
            switch (a[i]) {
            case 0:dna[i] = 'T'; break;
            case 1:dna[i] = 'C'; break;
            case 2:dna[i] = 'G'; break;
            case 3:dna[i] = 'A'; break;
            }
        }
    }
    else if (n == 7) {
        for (int i = 0; i < 4; i++) {
            switch (a[i]) {
            case 0:dna[i] = 'T'; break;
            case 1:dna[i] = 'G'; break;
            case 2:dna[i] = 'C'; break;
            case 3:dna[i] = 'A'; break;
            }
        }
    }
}


__device__ unsigned char dnaAdd(unsigned char a, unsigned char b)
{
    unsigned char result = 0;
    if (a == 'A') {
        switch (b) {
        case 'A':result = 'T'; break;
        case 'C':result = 'A'; break;
        case 'G':result = 'C'; break;
        case 'T':result = 'G'; break;
        }
    }
    else if (a == 'C') {
        switch (b) {
        case 'A':result = 'A'; break;
        case 'C':result = 'C'; break;
        case 'G':result = 'G'; break;
        case 'T':result = 'T'; break;
        }
    }
    else if (a == 'G') {
        switch (b) {
        case 'A':result = 'C'; break;
        case 'C':result = 'G'; break;
        case 'G':result = 'T'; break;
        case 'T':result = 'A'; break;
        }
    }
    else if (a == 'T') {
        switch (b) {
        case 'A':result = 'G'; break;
        case 'C':result = 'T'; break;
        case 'G':result = 'A'; break;
        case 'T':result = 'C'; break;
        }
    }
    return result;
}


__device__ unsigned char dnaDecode(unsigned char* dna, unsigned char n)
{
    unsigned char temp[4]{ 0,0,0,0 };
    if (n == 0) {
        for (int i = 0; i < 4; i++) {
            switch (dna[i]) {
            case 'C':temp[i] = 2; break;
            case 'A':temp[i] = 0; break;
            case 'T':temp[i] = 3; break;
            case 'G':temp[i] = 1; break;
            }
        }
    }
    else if (n == 1) {
        for (int i = 0; i < 4; i++) {
            switch (dna[i]) {
            case 'C':temp[i] = 1; break;
            case 'A':temp[i] = 0; break;
            case 'T':temp[i] = 3; break;
            case 'G':temp[i] = 2; break;
            }
        }
    }
    else if (n == 2) {
        for (int i = 0; i < 4; i++) {
            switch (dna[i]) {
            case 'C':temp[i] = 3; break;
            case 'A':temp[i] = 1; break;
            case 'T':temp[i] = 2; break;
            case 'G':temp[i] = 0; break;
            }
        }
    }
    else if (n == 3) {
        for (int i = 0; i < 4; i++) {
            switch (dna[i]) {
            case 'C':temp[i] = 0; break;
            case 'A':temp[i] = 1; break;
            case 'T':temp[i] = 2; break;
            case 'G':temp[i] = 3; break;
            }
        }
    }
    else if (n == 4) {
        for (int i = 0; i < 4; i++) {
            switch (dna[i]) {
            case 'C':temp[i] = 0; break;
            case 'A':temp[i] = 2; break;
            case 'T':temp[i] = 1; break;
            case 'G':temp[i] = 3; break;
            }
        }
    }
    else if (n == 5) {
        for (int i = 0; i < 4; i++) {
            switch (dna[i]) {
            case 'C':temp[i] = 3; break;
            case 'A':temp[i] = 2; break;
            case 'T':temp[i] = 1; break;
            case 'G':temp[i] = 0; break;
            }
        }
    }
    else if (n == 6) {
        for (int i = 0; i < 4; i++) {
            switch (dna[i]) {
            case 'C':temp[i] = 1; break;
            case 'A':temp[i] = 3; break;
            case 'T':temp[i] = 0; break;
            case 'G':temp[i] = 2; break;
            }
        }
    }
    else if (n == 7) {
        for (int i = 0; i < 4; i++) {
            switch (dna[i]) {
            case 'C':temp[i] = 2; break;
            case 'A':temp[i] = 3; break;
            case 'T':temp[i] = 0; break;
            case 'G':temp[i] = 1; break;
            }
        }
    }
    return temp[0] * 64 + temp[1] * 16 + temp[2] * 4 + temp[3];
}

__device__ unsigned char dnaMinus(unsigned char a, unsigned char b)
{
    //a-b
    unsigned char result = 0;
    if (a == 'A') {
        switch (b) {
            case 'A':result = 'C'; break;
            case 'C':result = 'A'; break;
            case 'G':result = 'T'; break;
            case 'T':result = 'G'; break;
        }
    }
    else if (a == 'C') {
        switch (b) {
            case 'A':result = 'G'; break;
            case 'C':result = 'C'; break;
            case 'G':result = 'A'; break;
            case 'T':result = 'T'; break;
        }
    }
    else if (a == 'G') {
        switch (b) {
            case 'A':result = 'T'; break;
            case 'C':result = 'G'; break;
            case 'G':result = 'C'; break;
            case 'T':result = 'A'; break;
        }
    }
    else if (a == 'T') {
        switch (b) {
            case 'A':result = 'A'; break;
            case 'C':result = 'T'; break;
            case 'G':result = 'G'; break;
            case 'T':result = 'C'; break;
        }
    }
    return result;
}
__device__ int dna_to_num(unsigned char* dna,int dna_rule)
{
    int result = 0;
    if (dna_rule == 0) {
        switch (dna[0]) {
            case 'A':result+=0; break;
            case 'C':result += 4; break;
            case 'G':result += 8; break;
            case 'T':result += 12; break;
        }
        switch (dna[1]) {
            case 'A':result += 0; break;
            case 'C':result += 1; break;
            case 'G':result += 2; break;
            case 'T':result += 3; break;
        }
    }
    else if (dna_rule == 1) {
        switch (dna[0]) {
            case 'A':result += 0; break;
            case 'G':result += 4; break;
            case 'C':result += 8; break;
            case 'T':result += 12; break;
        }
        switch (dna[1]) {
            case 'A':result += 0; break;
            case 'G':result += 1; break;
            case 'C':result += 2; break;
            case 'T':result += 3; break;
        }
    }
    else if (dna_rule == 2) {
        switch (dna[0]) {
            case 'C':result += 0; break;
            case 'A':result += 4; break;
            case 'T':result += 8; break;
            case 'G':result += 12; break;
        }
        switch (dna[1]) {
            case 'C':result += 0; break;
            case 'A':result += 1; break;
            case 'T':result += 2; break;
            case 'G':result += 3; break;
        }
    }
    else if (dna_rule == 3) {
        switch (dna[0]) {
            case 'G':result += 0; break;
            case 'A':result += 4; break;
            case 'T':result += 8; break;
            case 'C':result += 12; break;
        }
        switch (dna[1]) {
            case 'G':result += 0; break;
            case 'A':result += 1; break;
            case 'T':result += 2; break;
            case 'C':result += 3; break;
        }
    }
    else if (dna_rule == 4) {
        switch (dna[0]) {
            case 'C':result += 0; break;
            case 'T':result += 4; break;
            case 'A':result += 8; break;
            case 'G':result += 12; break;
        }
        switch (dna[1]) {
            case 'C':result += 0; break;
            case 'T':result += 1; break;
            case 'A':result += 2; break;
            case 'G':result += 3; break;
        }
    }
    else if (dna_rule == 5) {
        switch (dna[0]) {
            case 'G':result += 0; break;
            case 'T':result += 4; break;
            case 'A':result += 8; break;
            case 'C':result += 12; break;
        }
        switch (dna[1]) {
            case 'G':result += 0; break;
            case 'T':result += 1; break;
            case 'A':result += 2; break;
            case 'C':result += 3; break;
        }
    }
    else if (dna_rule == 6) {
        switch (dna[0]) {
            case 'T':result += 0; break;
            case 'C':result += 4; break;
            case 'G':result += 8; break;
            case 'A':result += 12; break;
        }
        switch (dna[1]) {
            case 'T':result += 0; break;
            case 'C':result += 1; break;
            case 'G':result += 2; break;
            case 'A':result += 3; break;
        }
    }
    else if (dna_rule == 7) {
        switch (dna[0]) {
            case 'T':result += 0; break;
            case 'G':result += 4; break;
            case 'C':result += 8; break;
            case 'A':result += 12; break;
        }
        switch (dna[1]) {
            case 'T':result += 0; break;
            case 'G':result += 1; break;
            case 'C':result += 2; break;
            case 'A':result += 3; break;
        }
    }
    return result;
}

__device__ char num_to_dna(int num, int dna_rule)
{
    char result = 0;
    if (dna_rule == 0) {
        switch (num) {
            case 0:result = 'A'; break;
            case 1:result = 'C'; break;
            case 2:result = 'G'; break;
            case 3:result = 'T'; break;
        }
    }
    else if (dna_rule == 1) {
        switch (num) {
            case 0:result = 'A'; break;
            case 1:result = 'G'; break;
            case 2:result = 'C'; break;
            case 3:result = 'T'; break;
        }
    }
    else if (dna_rule == 2) {
        switch (num) {
            case 0:result = 'C'; break;
            case 1:result = 'A'; break;
            case 2:result = 'T'; break;
            case 3:result = 'G'; break;
        }
    }
    else if (dna_rule == 3) {
        switch (num) {
            case 0:result = 'G'; break;
            case 1:result = 'A'; break;
            case 2:result = 'T'; break;
            case 3:result = 'C'; break;
        }
    }
    else if (dna_rule == 4) {
        switch (num) {
            case 0:result = 'C'; break;
            case 1:result = 'T'; break;
            case 2:result = 'A'; break;
            case 3:result = 'G'; break;
        }
    }
    else if (dna_rule == 5) {
        switch (num) {
            case 0:result = 'G'; break;
            case 1:result = 'T'; break;
            case 2:result = 'A'; break;
            case 3:result = 'C'; break;
        }
    }
    else if (dna_rule == 6) {
        switch (num) {
            case 0:result = 'T'; break;
            case 1:result = 'C'; break;
            case 2:result = 'G'; break;
            case 3:result = 'A'; break;
        }
    }
    else if (dna_rule == 7) {
        switch (num) {
            case 0:result = 'T'; break;
            case 1:result = 'G'; break;
            case 2:result = 'C'; break;
            case 3:result = 'A'; break;
        }
    }
    return result;
}