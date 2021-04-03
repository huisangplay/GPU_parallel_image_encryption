#include "compute.cuh"
#include<iostream>
using namespace std;
void imgRowExchange(Mat& img, int* p)
{
    for (int i = 0; i < img.rows; i++) {
        Mat temp1 = img.row(i).clone();
        Mat temp2 = img.row(p[i]).clone();

        temp1.copyTo(img.row(p[i]));
        temp2.copyTo(img.row(i));
    }
}

void imgColExchange(Mat& img, int* q)
{
    for (int j = 0; j < img.cols; j++) {
        Mat temp1 = img.col(j).clone();
        Mat temp2 = img.col(q[j]).clone();
        temp1.copyTo(img.col(q[j]));
        temp2.copyTo(img.col(j));
    }
}

void imgConfusion(Mat& img)
{
    //获取2D-HSM混沌序列
    int n = img.rows+img.cols;
    double* temp_X = new double[n], * temp_Y = new double[n];
    int* row_X = new int[n], * row_Y = new int[n];
    getSine2DSequence(temp_X, temp_Y, n);

    for (int i = 0; i < n; i++) {
        row_X[i] = static_cast<long long>(temp_X[i] * 1e14)% img.rows;
        row_Y[i] = static_cast<long long>(temp_Y[i] * 1e14)% img.rows;
        if (row_X[i] < 0)  row_X[i] = -row_X[i];
        if (row_Y[i] < 0)  row_Y[i] = -row_Y[i];
    }

    int* col_X = new int[n], * col_Y = new int[n];
    getSine2DSequence(temp_X, temp_Y, n);
    for (int i = 0; i < n; i++) {
        col_X[i] = static_cast<long long>(temp_X[i] * 1e14) % img.cols;
        col_Y[i] = static_cast<long long>(temp_Y[i] * 1e14) % img.cols;

        if (col_X[i] < 0)  col_X[i] = -col_X[i];
        if (col_Y[i] < 0)  col_Y[i] = -col_Y[i];
    }
    delete[] temp_X;
    delete[] temp_Y;

    for(int i=0;i<n;i++){
        //交换行
        int index=i%img.rows;
        Mat temp1 = img.row(index).clone();
        Mat temp2 = img.row(row_X[i]).clone();

        temp1.copyTo(img.row(row_X[i]));
        temp2.copyTo(img.row(index));

        //交换列
        index=i%img.cols;
        Mat temp3 = img.col(index).clone();
        Mat temp4 = img.col(col_X[i]).clone();
        temp3.copyTo(img.col(col_X[i]));
        temp4.copyTo(img.col(index));
    }

    for(int i=0;i<n;i++){
        //交换行
        int index=i%img.rows;
        Mat temp1 = img.row(index).clone();
        Mat temp2 = img.row(row_Y[i]).clone();

        temp1.copyTo(img.row(row_Y[i]));
        temp2.copyTo(img.row(index));

        //交换列
        index=i%img.cols;
        Mat temp3 = img.col(index).clone();
        Mat temp4 = img.col(col_Y[i]).clone();
        temp3.copyTo(img.col(col_Y[i]));
        temp4.copyTo(img.col(index));
    }

    delete[] row_X;
    delete[] row_Y;
    delete[] col_X;
    delete[] col_Y;
}

void chaosSelectByCol(int* chaos, double* update, int n)
{
    getLogisticSequence(update, n);
    for (int i = 0; i < n; i++) {
        chaos[i] = static_cast<long long>(update[i] * 1e14) % 3;
    }
}

void updateKeys(unsigned char buf[])
{

    //更新密钥
    double CK = std_a + std_b+std_k + cat3D_x + cat3D_y + cat3D_z + lorenz_y + lorenz_z + lorenz_q + sine2D_a + sine2D_b + sine2D_x + sine2D_y + logistic_x;
    CK = fmod(CK, 1);

    std_a = std_a + ((static_cast<long long>(buf[0]) << 8) + buf[1]) / 65536.0 + CK; std_a = fmod(std_a, 2*pi);
    std_b = std_b + ((static_cast<long long>(buf[2]) << 8) + buf[3] ) / 65536.0 + CK; std_b = fmod(std_b, 2*pi);
    std_k = std_k + ((static_cast<long long>(buf[4]) << 8) + buf[5]) / 65536.0 + CK; std_k =fmod(std_k, 1);

    cat3D_x = cat3D_x + ((static_cast<long long>(buf[6]) << 8) + buf[7] ) / 65536.0 + CK; cat3D_x =fmod(cat3D_x, 1);
    cat3D_y = cat3D_y + ((static_cast<long long>(buf[8]) << 8) + buf[9] ) / 65536.0 + CK; cat3D_y= fmod(cat3D_y, 1);
    cat3D_z = cat3D_z + ((static_cast<long long>(buf[10]) << 8) + buf[11] ) / 65536.0 + CK; cat3D_z=fmod(cat3D_z, 1);

    lorenz_y = lorenz_y + ((static_cast<long long>(buf[12]) << 8) + buf[13] ) / 65536.0 + CK; lorenz_y=fmod(lorenz_y, 1);
    lorenz_z = lorenz_z + ((static_cast<long long>(buf[14]) << 8) + buf[15] ) / 65536.0 + CK; lorenz_z=fmod(lorenz_z, 1);
    lorenz_q = lorenz_q + ((static_cast<long long>(buf[16]) << 8) + buf[17] ) / 65536.0 + CK; lorenz_q=fmod(lorenz_q, 1);

    logistic_x = logistic_x + ((static_cast<long long>(buf[18]) << 8) + buf[19]) / 65536.0 + CK; logistic_x = fmod(logistic_x, 1);

    sine2D_a = sine2D_a + ((static_cast<long long>(buf[20]) << 8)+(static_cast<long long>(buf[21]) << 8)
        + buf[22] ) / 65536.0/256.0 + CK; 
    sine2D_b = sine2D_b + ((static_cast<long long>(buf[23]) << 8) + (static_cast<long long>(buf[24]) << 8)
        + buf[25]) / 65536.0 / 256.0 + CK;
    sine2D_x = sine2D_x + ((static_cast<long long>(buf[26]) << 8) + (static_cast<long long>(buf[27]) << 8)
        + buf[28]) / 65536.0 / 256.0 + CK;
    sine2D_x=fmod(sine2D_x, 1);
    sine2D_y = sine2D_y + ((static_cast<long long>(buf[29]) << 8) + (static_cast<long long>(buf[30]) << 8)
        + buf[31]) / 65536.0 / 256.0 + CK;
    sine2D_y = fmod(sine2D_y, 1);
    /*
    cout << sine_x << "--" << sine_u << "--" << cat3D_x << "--" <<cat3D_y << "--" << cat3D_z
        << "--" << lorenz_y << "--" << lorenz_z << "--" << lorenz_q << "--" << sine2D_a << "--" << sine2D_b
        << "--" << sine2D_x << "--" << sine2D_y << "--" <<logistic_u << "--" << logistic_x<<endl;
    */

    cudaMemcpyToSymbol(dev_std_a, &std_a, sizeof(double));
    cudaMemcpyToSymbol(dev_std_b, &std_b, sizeof(double));
    cudaMemcpyToSymbol(dev_std_k, &std_k, sizeof(double));
    cudaMemcpyToSymbol(dev_cat3D_x, &cat3D_x, sizeof(double));
    cudaMemcpyToSymbol(dev_cat3D_y, &cat3D_y, sizeof(double));
    cudaMemcpyToSymbol(dev_cat3D_z, &cat3D_z, sizeof(double));
    cudaMemcpyToSymbol(dev_lorenz_y, &lorenz_y, sizeof(double));
    cudaMemcpyToSymbol(dev_lorenz_z, &lorenz_z, sizeof(double));
    cudaMemcpyToSymbol(dev_lorenz_q, &lorenz_q, sizeof(double));

}

void deImgConfusion(Mat& img)
{
    //获取2D-HSM混沌序列
    int n = img.rows+img.cols;
    double* temp_X = new double[n], * temp_Y = new double[n];
    int* row_X = new int[n], * row_Y = new int[n];
    getSine2DSequence(temp_X, temp_Y, n);

    for (int i = 0; i < n; i++) {
        row_X[i] = static_cast<long long>(temp_X[i] * 1e14)% img.rows;
        row_Y[i] = static_cast<long long>(temp_Y[i] * 1e14)% img.rows;
        if (row_X[i] < 0)  row_X[i] = -row_X[i];
        if (row_Y[i] < 0)  row_Y[i] = -row_Y[i];
    }

    int* col_X = new int[n], * col_Y = new int[n];
    getSine2DSequence(temp_X, temp_Y, n);
    for (int i = 0; i < n; i++) {
        col_X[i] = static_cast<long long>(temp_X[i] * 1e14) % img.cols;
        col_Y[i] = static_cast<long long>(temp_Y[i] * 1e14) % img.cols;

        if (col_X[i] < 0)  col_X[i] = -col_X[i];
        if (col_Y[i] < 0)  col_Y[i] = -col_Y[i];
    }
    delete[] temp_X;
    delete[] temp_Y;

    for(int i=n-1;i>=0;i--){
        //交换列
        int index=i%img.cols;
        Mat temp3 = img.col(index).clone();
        Mat temp4 = img.col(col_Y[i]).clone();
        temp3.copyTo(img.col(col_Y[i]));
        temp4.copyTo(img.col(index));


        //交换行
        index=i%img.rows;
        Mat temp1 = img.row(index).clone();
        Mat temp2 = img.row(row_Y[i]).clone();

        temp1.copyTo(img.row(row_Y[i]));
        temp2.copyTo(img.row(index));

    }


    for(int i=n-1;i>=0;i--){
        //交换列
        int index=i%img.cols;
        Mat temp3 = img.col(index).clone();
        Mat temp4 = img.col(col_X[i]).clone();
        temp3.copyTo(img.col(col_X[i]));
        temp4.copyTo(img.col(index));

        //交换行
        index=i%img.rows;
        Mat temp1 = img.row(index).clone();
        Mat temp2 = img.row(row_X[i]).clone();

        temp1.copyTo(img.row(row_X[i]));
        temp2.copyTo(img.row(index));
    }

    delete[] row_X;
    delete[] row_Y;
    delete[] col_X;
    delete[] col_Y;
}

void deImgColExchange(Mat &img, int *q) {
    for (int j = img.cols-1; j >=0; j--) {
        Mat temp1 = img.col(j).clone();
        Mat temp2 = img.col(q[j]).clone();
        temp1.copyTo(img.col(q[j]));
        temp2.copyTo(img.col(j));
    }
}
void deImgRowExchange(Mat& img, int* p)
{
    for (int i = img.rows-1; i >=0; i--) {
        Mat temp1 = img.row(i).clone();
        Mat temp2 = img.row(p[i]).clone();

        temp1.copyTo(img.row(p[i]));
        temp2.copyTo(img.row(i));
    }
}