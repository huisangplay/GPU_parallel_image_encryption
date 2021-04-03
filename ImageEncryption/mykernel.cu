#include"mykernel.cuh"
#include<stdio.h>
__global__ void dnaDecodeKernel(unsigned char* dev_dna, unsigned char* dev_img, unsigned char* dev_dnaProSequence, int rows,int cols) {
    int j = blockIdx.x*128+threadIdx.x;
    if(j>=cols) return;
    //i行j列
    int i=blockIdx.y;
    dev_img[cols * i + j] = dnaDecode(&dev_dna[i * cols * 4 + j*4], dev_dnaProSequence[j * rows * 10 + i + rows * 9]);
}
__global__ void getDnaAddSequenceKernel(unsigned char* dev_dnaAdd, unsigned char* dev_dnaProSequence, int rows, int cols)
{
    int j = blockIdx.x*128+threadIdx.x;
    if(j>=cols) return;
    //i行j列
    for (int i = 0; i < rows; i++) {
        //这里的dna编码规则为随机生成的规则，为rows*8~rows*9
        dnaEncode(dev_dnaProSequence[j * rows * 10 + i + rows * 7], &dev_dnaAdd[i * cols * 4 + j * 4], dev_dnaProSequence[j * rows * 10 + i + rows * 8]);
    }
}

__global__ void dnaXorKernel(unsigned char* dev_dna, unsigned char* dev_dnaXor, int size) {
    int index=blockIdx.x*128+threadIdx.x;
    if(index>=size) return;
    dev_dna[index]=dnaXor(dev_dna[index],dev_dnaXor[index]);
}

__global__ void dnaAddKernel(unsigned char* dev_dna, unsigned char* dev_dnaAdd, int size)
{
    int index=blockIdx.x*128+threadIdx.x;
    if(index>=size) return;
    dev_dna[index] = dnaAdd(dev_dna[index], dev_dnaAdd[index]);
}

__global__ void getDnaXorSequenceKernel(unsigned char* dev_dnaXor, unsigned char* dev_dnaProSequence, int rows,int cols) {
    int j = blockIdx.x*128+threadIdx.x;
    if(j>=cols) return;
    //i行j列
    for (int i = 0; i < rows; i++) {
        //这里的dna编码规则为随机生成的规则，为rows*6~rows*7
        dnaEncode(dev_dnaProSequence[j * rows * 10 + i+rows*5], &dev_dnaXor[i*cols*4+j*4], dev_dnaProSequence[j * rows * 10 + i+rows*6]);
    }
}

__global__ void test(unsigned char* dev_dna, int n) {
    for (int i = n; i < 50+n; i++) {
        printf("%c",dev_dna[i]);
    }
    printf("\n");

}
__global__ void test(int* seq, int n) {
    for (int i = n; i < 50+n; i++) {
        printf("%d",seq[i]);
    }
    printf("\n");
}


__global__ void  imgDnaEncodeKernel(unsigned char* dev_img, unsigned char* dev_dna, unsigned char* dev_dnaProSequence, int rows,int cols) {
    int j = blockIdx.x*128+threadIdx.x;
    if(j>=cols) return;
    int i=blockIdx.y;
    //i行j列
    dnaEncode(dev_img[cols * i + j], &dev_dna[cols * i * 4 + j * 4], dev_dnaProSequence[j * rows * 10 + i]);
}


__global__ void getDnaProSequenceKernel(unsigned char* dev_dnaProSequence, double* dev_update, int* dev_chaos, int rows,int cols)
{
    int i = blockIdx.x*128+threadIdx.x;

    if(i>=cols) return;
    int t = 1000;

    if (dev_chaos[i] == 0) {
        double a = dev_std_a + dev_update[i];
        double b = dev_std_b+ (static_cast<double>(dev_update[i] * 1e5) - static_cast<long long>(dev_update[i] * 1e5));
        double k = dev_std_k+ (static_cast<double>(dev_update[i] * 1e10) - static_cast<long long>(dev_update[i] * 1e10));
        for (int j = 0; j < t; j++) {
            double temp_a = a + b;
            while (temp_a < 0) temp_a += 2 * dev_pi;
            temp_a = fmod(temp_a, 2 * dev_pi);
            double temp_b = b + k * sin(a + b);
            while (temp_b < 0) temp_b += 2 * dev_pi;
            temp_b = fmod(temp_b, 2 * dev_pi);

            a = temp_a;
            b = temp_b;
        }
        for (int j = 0; j < rows * 10; j++) {
            double temp_a = a + b;
            while (temp_a < 0) temp_a += 2 * dev_pi;
            temp_a = fmod(temp_a, 2 * dev_pi);
            double temp_b = b + k * sin(a + b);
            while (temp_b < 0) temp_b += 2 * dev_pi;
            temp_b = fmod(temp_b, 2 * dev_pi);

            a = temp_a;
            b = temp_b;
            long long sum = static_cast<long long>((a+b)/2 * 1e14);
            long index = rows * i * 10 + j;
            if (j < rows && j >= 0) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows && j < rows * 3) {
                dev_dnaProSequence[index] = sum % 4;
            }
            else if (j >= rows * 3 && j < rows * 5) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 5 && j < rows * 6) {
                dev_dnaProSequence[index] = sum % 256;
            }
            else if (j >= rows * 6 && j < rows * 7) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 7 && j < rows * 8) {
                dev_dnaProSequence[index] = sum % 256;
            }
            else if (j >= rows * 8 && j < rows * 9) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 9 && j < rows * 10) {
                dev_dnaProSequence[index] = sum % 8;
            }
        }
    }
    else if (dev_chaos[i] == 1) {
        double x = dev_cat3D_x + dev_update[i]; x = fmod(x, 1.0);
        double y = dev_cat3D_x + (static_cast<double>(dev_update[i] * 1e5) - static_cast<long long>(dev_update[i] * 1e5)); y = fmod(y, 1.0);
        double z = dev_cat3D_z + (static_cast<double>(dev_update[i] * 1e10) - static_cast<long long>(dev_update[i] * 1e10)); z = fmod(z, 1.0);

        double temp_x = x;
        double temp_y = y;
        double temp_z = z;


        for (long j = 0; j < t; ++j) {
            temp_x = 2 * x + y + 3 * z;
            temp_x = temp_x - (int)temp_x;
            temp_y = 3 * x + 2 * y + 5 * z;
            temp_y = temp_y - (int)temp_y;
            temp_z = 2 * x + y + 4 * z;
            temp_z = temp_z - (int)temp_z;

            x = temp_x;
            y = temp_y;
            z = temp_z;
        }

        for (long j = 0; j < rows * 10; ++j) {
            temp_x = 2 * x + y + 3 * z;
            temp_x = temp_x - (int)temp_x;
            temp_y = 3 * x + 2 * y + 5 * z;
            temp_y = temp_y - (int)temp_y;
            temp_z = 2 * x + y + 4 * z;
            temp_z = temp_z - (int)temp_z;

            x = temp_x;
            y = temp_y;
            z = temp_z;
            long long sum = (static_cast<long long>((x + y + z) / 3 * 1e14));
            long index = rows * i * 10 + j;
            if (j < rows && j >= 0) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows && j < rows * 3) {
                dev_dnaProSequence[index] = sum % 4;
            }
            else if (j >= rows * 3 && j < rows * 5) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 5 && j < rows * 6) {
                dev_dnaProSequence[index] = sum % 256;
            }
            else if (j >= rows * 6 && j < rows * 7) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 7 && j < rows * 8) {
                dev_dnaProSequence[index] = sum % 256;
            }
            else if (j >= rows * 8 && j < rows * 9) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 9 && j < rows * 10) {
                dev_dnaProSequence[index] = sum % 8;
            }

        }
    }
    else if (dev_chaos[i] == 2) {
        double y = dev_lorenz_y + dev_update[i];
        double z = dev_lorenz_z + (static_cast<double>(dev_update[i] * 1e5) - static_cast<long long>(dev_update[i] * 1e5));
        double q = dev_lorenz_q + (static_cast<double>(dev_update[i] * 1e10) - static_cast<long long>(dev_update[i] * 1e10));
        //double h = 0.005;
        double h = 0.005;
        for (int j = 0; j < t; j++) {
            double y1 = -dev_lorenz_f * y + dev_lorenz_f * z;
            double z1 = dev_lorenz_r * y - z - y * q;
            double q1 = -dev_lorenz_g * q + y * z;

            double y2 = -dev_lorenz_f * (y + h / 2 * y1) + dev_lorenz_f * (z + h / 2 * z1);
            double z2 = dev_lorenz_r * (y + h / 2 * y1) - (z + h / 2 * z1) - (y + h / 2 * y1) * (q + h / 2 * q1);
            double q2 = -dev_lorenz_g * (q + h / 2 * q1) + (y + h / 2 * y1) * (z + h / 2 * z1);

            double y3 = -dev_lorenz_f * (y + h / 2 * y2) + dev_lorenz_f * (z + h / 2 * z2);
            double z3 = dev_lorenz_r * (y + h / 2 * y2) - (z + h / 2 * z2) - (y + h / 2 * y2) * (q + h / 2 * q2);
            double q3 = -dev_lorenz_g * (q + h / 2 * q2) + (y + h / 2 * y2) * (z + h / 2 * z2);

            double y4 = -dev_lorenz_f * (y + h * y3) + dev_lorenz_f * (z + h * z3);
            double z4 = dev_lorenz_r * (y + h * y3) - (z + h * z3) - (y + h * y3) * (q + h * q3);
            double q4 = -dev_lorenz_g * (q + h * q3) + (y + h * y3) * (z + h * z3);

            double temp_y = y + h / 6 * (y1 + 2 * y2 + 2 * y3 + y4);
            double temp_z = z + h / 6 * (z1 + 2 * z2 + 2 * z3 + z4);
            double temp_q = q + h / 6 * (q1 + 2 * q2 + 2 * q3 + q4);

            y = temp_y;
            z = temp_z;
            q = temp_q;
        }
        for (int j = 0; j < rows * 10; j++) {
            double y1 = -dev_lorenz_f * y + dev_lorenz_f * z;
            double z1 = dev_lorenz_r * y - z - y * q;
            double q1 = -dev_lorenz_g * q + y * z;

            double y2 = -dev_lorenz_f * (y + h / 2 * y1) + dev_lorenz_f * (z + h / 2 * z1);
            double z2 = dev_lorenz_r * (y + h / 2 * y1) - (z + h / 2 * z1) - (y + h / 2 * y1) * (q + h / 2 * q1);
            double q2 = -dev_lorenz_g * (q + h / 2 * q1) + (y + h / 2 * y1) * (z + h / 2 * z1);

            double y3 = -dev_lorenz_f * (y + h / 2 * y2) + dev_lorenz_f * (z + h / 2 * z2);
            double z3 = dev_lorenz_r * (y + h / 2 * y2) - (z + h / 2 * z2) - (y + h / 2 * y2) * (q + h / 2 * q2);
            double q3 = -dev_lorenz_g * (q + h / 2 * q2) + (y + h / 2 * y2) * (z + h / 2 * z2);

            double y4 = -dev_lorenz_f * (y + h * y3) + dev_lorenz_f * (z + h * z3);
            double z4 = dev_lorenz_r * (y + h * y3) - (z + h * z3) - (y + h * y3) * (q + h * q3);
            double q4 = -dev_lorenz_g * (q + h * q3) + (y + h * y3) * (z + h * z3);

            double temp_y = y + h / 6 * (y1 + 2 * y2 + 2 * y3 + y4);
            double temp_z = z + h / 6 * (z1 + 2 * z2 + 2 * z3 + z4);
            double temp_q = q + h / 6 * (q1 + 2 * q2 + 2 * q3 + q4);

            y = temp_y;
            z = temp_z;
            q = temp_q;

            long long sum = static_cast<long long>(fabs(q + y + z) / 3 * 1e14);//由于洛伦兹混沌系统产生的值会有负值，所以需要求绝对值
            long index = rows * i * 10 + j;
            if (j < rows && j >= 0) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows && j < rows * 3) {
                dev_dnaProSequence[index] = sum % 4;
            }
            else if (j >= rows * 3 && j < rows * 5) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 5 && j < rows * 6) {
                dev_dnaProSequence[index] = sum % 256;
            }
            else if (j >= rows * 6 && j < rows * 7) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 7 && j < rows * 8) {
                dev_dnaProSequence[index] = sum % 256;
            }
            else if (j >= rows * 8 && j < rows * 9) {
                dev_dnaProSequence[index] = sum % 8;
            }
            else if (j >= rows * 9 && j < rows * 10) {
                dev_dnaProSequence[index] = sum % 8;
            }
        }
    }
}
__global__ void de_imgDnaEncodeKernel(unsigned char *dev_img, unsigned char *dev_dna, unsigned char* dev_dnaProSequence, int rows, int cols) {
    int j = blockIdx.x*128+threadIdx.x;
    if(j>=cols) return;
    //i行j列
    int i=blockIdx.y;
    dnaEncode(dev_img[cols * i + j], &dev_dna[cols * i * 4 + j * 4], dev_dnaProSequence[j * rows * 10 + i+rows*9]);
}
__global__ void dnaMinusKernel(unsigned char* dev_dna, unsigned char* dev_dnaMinus, int cols, int size)
{
    int index=blockIdx.x*128+threadIdx.x;
    if(index>=size) return;
    //dev_dna[blockIdx.x * cols + blockIdx.y] = dnaMinus(dev_dna[blockIdx.x * cols + blockIdx.y], dev_dnaMinus[blockIdx.x * cols + blockIdx.y]);
    dev_dna[index] = dnaMinus(dev_dna[index], dev_dnaMinus[index]);
}


__global__ void de_dnaDecodeKernel(unsigned char* dev_dna, unsigned char* dev_img, unsigned char* dev_dnaProSequence, int rows,int cols) {

    int j = blockIdx.x*128+threadIdx.x;
    if(j>=cols) return;
    //i行j列
    int i=blockIdx.y;
    dev_img[cols * i + j] = dnaDecode(&dev_dna[i * cols * 4 + j*4], dev_dnaProSequence[j * rows * 10 + i]);

}

__global__ void dna_sbox_replaceKernel(unsigned char* dev_dna, unsigned char* dev_dnaProSequence, int rows, int cols) {

    int j = blockIdx.x * 128 + threadIdx.x;
    if (j >= cols) return;
    int i = blockIdx.y;

    //ŒÆËãdna-sºÐÌæ»»ÐÐºÅ
    int row_index = dev_dnaProSequence[j * rows * 10 + i + rows];
    int dna_rule = dev_dnaProSequence[j * rows * 10 + i + rows * 3];
    char row_char = num_to_dna(row_index, dna_rule);

    switch (row_char) {
        case 'A':row_index = 0; break;
        case 'C':row_index = 1; break;
        case 'G':row_index = 2; break;
        case 'T':row_index = 3; break;
    }
    int col_index = dna_to_num(&dev_dna[i * cols * 2 + j * 2], dna_rule);
    dev_dna[i * cols * 2 + j * 2] = dna_sbox[dna_rule * 128 + row_index * 32 + col_index*2];
    dev_dna[i * cols * 2 + j * 2+1] = dna_sbox[dna_rule * 128 + row_index * 32 + col_index*2+1];
}
__global__ void de_dna_sbox_replaceKernel(unsigned char* dev_dna, unsigned char* dev_dnaProSequence, int rows, int cols) {
    int j = blockIdx.x * 128 + threadIdx.x;
    if (j >= cols) return;
    int i = blockIdx.y;

    //ŒÆËãdna-sºÐÌæ»»ÐÐºÅ
    int row_index = dev_dnaProSequence[j * rows * 10 + i + rows];
    int dna_rule = dev_dnaProSequence[j * rows * 10 + i + rows * 3];
    char row_char = num_to_dna(row_index, dna_rule);

    switch (row_char) {
        case 'A':row_index = 0; break;
        case 'C':row_index = 1; break;
        case 'G':row_index = 2; break;
        case 'T':row_index = 3; break;
    }

    for (int k = dna_rule * 128 + row_index * 32; k < (dna_rule * 128 + row_index * 32+32); k+=2) {
        if (dev_dna[i * cols * 2 + j * 2] == dna_sbox[k] && dev_dna[i * cols * 2 + j * 2 + 1] == dna_sbox[k + 1]) {
            int n=(k - (dna_rule * 128) - (row_index * 32))/2;
            dev_dna[i * cols * 2 + j * 2]= num_to_dna(n/4,dna_rule);
            dev_dna[i * cols * 2 + j * 2+1]= num_to_dna(n%4, dna_rule);
            break;
        }
    }
}