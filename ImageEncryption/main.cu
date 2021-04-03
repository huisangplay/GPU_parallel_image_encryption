#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include"compute.cuh"
#include<ctime>
#include<cstdlib>
#include <fstream>
#include"mykernel.cuh"
#include"analysis.cuh"
using namespace cv;
void imgEncryption(Mat& img,unsigned char buf[]);
void reSetKey();
void chaosPrePro();
Mat imgDecryption(Mat &cipher,unsigned char buf[]);
void run();
void imgChangeEncryption(char const *imgUrl);
int main()
{

    for(int i=0;i<200;i++){
        run();
        //system("cd /home/xiaozhou/Documents/MATLAB/test && /home/xiaozhou/application/matlab2019b/bin/matlab -nodesktop -nosplash -r main");
        printf("\n-------------------------------\n%d-----------------------------------\n",i);
    }
    return 0;
}

void run(){
    clock_t encryption_start, encryption_end,decryption_start,decryption_end;
    char  const   *imgUrl = "/home/xiaozhou/Pictures/wash-ir.tiff";
    Mat img = imread(imgUrl,-1);
    if (img.empty())
    {
        printf("图片读取失败...");
        return;
    }

    //计算图像的sha256哈希值
    unsigned char buf[SHA256_BLOCK_SIZE];
    SHA256_CTX ctx;

    sha256_init(&ctx);
    sha256_update(&ctx, img.data, (unsigned long long)img.rows * img.cols * img.channels());
    sha256_final(&ctx, buf);

    imwrite("/home/xiaozhou/Documents/MATLAB/test/lena.bmp", img);

    encryption_start = clock();
    imgEncryption(img,buf);//加密算法
    encryption_end = clock();//计时结束

    imwrite("/home/xiaozhou/Documents/MATLAB/test/lena_result.bmp", img);

    decryption_start=clock();
    Mat decryption_img=imgDecryption(img,buf);//解密算法
    decryption_end=clock();

    imgChangeEncryption(imgUrl);
    printf("color image encryption time:%f s\n", ((double)encryption_end - encryption_start) / CLOCKS_PER_SEC);
    printf("color image decryption time:%f s\n", ((double)decryption_end - decryption_start) / CLOCKS_PER_SEC);

    printf("卡方检验的值:%f\n",getChiSquareTests(img));

    std::ofstream ofs;
    double time1=((double)encryption_end - encryption_start)/CLOCKS_PER_SEC*0.75;
    double time2=((double)decryption_end - decryption_start)/CLOCKS_PER_SEC*0.75;
    ofs.open("encryption_time.txt", std::ios::app);
    ofs <<time1<< std::endl;
    ofs.close();

    ofs.open("decryption_time.txt", std::ios::app);
    ofs <<time2<<std::endl;
    ofs.close();
}

Mat imgDecryption(Mat &cipher,unsigned char buf[]){
    reSetKey();
    updateKeys(buf);
    chaosPrePro();

    int channels = cipher.channels();
    int rows = cipher.rows;
    int cols = cipher.cols * channels;
    Mat img = cipher.clone();
    //如果图像通道数大于1，则根据图像通道数进行分割
    if (channels > 1) {
        Mat* channel = new Mat[channels];
        split(cipher, channel);
        //将三个通道重新排列，保存到一个图像中，图像大小为row*3*col
        for (int i = 0; i < channels - 1; i++) {
            hconcat(channel[i], channel[i + 1], channel[i + 1]);
        }
        img = channel[channels - 1].clone();
        delete[] channel;
    }

    //cuda部分
    cudaError_t cudaStatus;

    unsigned char* dev_img = 0;
    cudaStatus = cudaMalloc((void**)&dev_img, static_cast<long long>(rows)* cols * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_img cudaMalloc failed!\n");
    }
    cudaStatus = cudaMemcpy(dev_img, img.data, static_cast<long long>(rows)* cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_img cudaMemcpy failed!\n");
    }

    /*
    chaos用于每一列的混沌序列选择
    update用于更新每一列使用的混沌序列的初始密钥
    */
    int* chaos = new int[cols + rows];
    double* update = new double[cols + rows];
    chaosSelectByCol(chaos, update, cols + rows);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    //dna加密图像所用的序列
    unsigned char* dev_dnaProSequence;
    cudaStatus = cudaMalloc((void**)&dev_dnaProSequence, static_cast<long long>(rows)* cols * 10 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, " dev_dnaProSequence cudaMalloc failed!");
    }

    double* dev_update;
    cudaStatus = cudaMalloc((void**)&dev_update, (rows + cols) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_update cudaMalloc failed!\n");
    }
    cudaStatus = cudaMemcpy(dev_update, update, (rows + cols) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_update cudaMemcpy failed!\n");
    }


    int* dev_chaos;
    cudaStatus = cudaMalloc((void**)&dev_chaos, (rows + cols) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_chaos cudaMalloc failed!\n");
    }
    cudaStatus = cudaMemcpy(dev_chaos, chaos, (rows + cols) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_chaos cudaMemcpy failed!\n");
    }


    //获取每一列所使用的混沌序列
    getDnaProSequenceKernel <<<(cols+128-1)/128, 128>>> (dev_dnaProSequence, dev_update, dev_chaos, rows,cols);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "getDnaEncodeRulesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "getDnaEncodeRulesKernel cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    //dna编码
    uchar* dev_dna = 0;
    cudaStatus = cudaMalloc((void**)&dev_dna, static_cast<long long>(rows)* cols * 4 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_dna cudaMalloc failed!\n");
    }

    de_imgDnaEncodeKernel <<<dim3((cols+128-1)/128,rows), 128>>> (dev_img, dev_dna, dev_dnaProSequence, rows, cols);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "imgDnaEncodeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching imgDnaEncodeKernel!\n", cudaStatus);
    }

    //生成减操作序列
    unsigned char* dev_dnaAdd;
    cudaStatus = cudaMalloc((void**)&dev_dnaAdd, static_cast<long long>(rows)* cols * 4 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_dnaXor cudaMalloc failed!\n");
    }
    getDnaAddSequenceKernel <<<(cols+128-1)/128, 128 >>> (dev_dnaAdd, dev_dnaProSequence, rows, cols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaXorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaXorKernel!\n", cudaStatus);
    }

    dnaMinusKernel <<<(rows*cols*4+128-1)/128, 128  >>> (dev_dna, dev_dnaAdd, cols,rows*cols*4);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaXorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaXorKernel!\n", cudaStatus);
    }

    //将随机序列编码成dna序列，用于dna异或操作
    unsigned char* dev_dnaXor;
    cudaStatus = cudaMalloc((void**)&dev_dnaXor, static_cast<long long>(rows)* cols * 4 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_dnaXor cudaMalloc failed!\n");
    }
    getDnaXorSequenceKernel <<<(cols+128-1)/128, 128 >>> (dev_dnaXor, dev_dnaProSequence, rows, cols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaXorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaXorKernel!\n", cudaStatus);
    }

    //将随机dna序列与明文图像的dna进行异或操作
    dnaXorKernel <<<(rows*cols*4+128-1)/128, 128 >>> (dev_dna, dev_dnaXor, rows*cols*4);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaXorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaXorKernel!\n", cudaStatus);
    }



    //dna编码在列内交换位置逆过程
    //DNA-S盒替换逆运算
    de_dna_sbox_replaceKernel << <dim3((cols + 128 - 1) / 128, rows * 2), 128 >> > (dev_dna, dev_dnaProSequence, rows, cols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "de_dna_sbox_replaceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching de_dna_sbox_replaceKernel!\n", cudaStatus);
    }


    //dna解码
    de_dnaDecodeKernel <<<dim3((cols+128-1)/128,rows), 128 >>> (dev_dna, dev_img, dev_dnaProSequence, rows, cols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaDecodeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaDecodeKernel!\n", cudaStatus);
    }
    cudaStatus=cudaMemcpy(img.data, dev_img, sizeof(unsigned char) * static_cast<long long>(rows)* cols, cudaMemcpyDeviceToHost);
    if(cudaStatus!= cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus= cudaDeviceReset();
    if(cudaStatus!= cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    deImgConfusion(img);

    Mat final;

    if (channels > 1) {
        //分割图像然后通道合并
        Mat* channel=new Mat[3];
        channel[0]=Mat(img,Rect(0,0,cipher.cols,cipher.rows));
        channel[1]=Mat(img,Rect(cipher.cols,0,cipher.cols,cipher.rows));
        channel[2]=Mat(img,Rect(cipher.cols*2,0,cipher.cols,cipher.rows));

        merge(channel,3,final);

        delete[] channel;
    }
    else {
        final=img.clone();
    }
    return final;
}

void imgChangeEncryption(char const *imgUrl){
    Mat imgChange = imread(imgUrl,-1);

    unsigned seed;
    seed = time(0);
    srand(seed);
    unsigned rowChange = rand() % imgChange.rows;
    unsigned colChange = rand() % imgChange.cols;
    unsigned valuePix = rand() % 256;
    imgChange.data[rowChange * imgChange.cols + colChange] = (imgChange.data[rowChange * imgChange.cols + colChange] + valuePix) % 256;


    unsigned char buf[SHA256_BLOCK_SIZE];
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, imgChange.data, (unsigned long long)imgChange.rows * imgChange.cols * imgChange.channels());
    sha256_final(&ctx, buf);

    imgEncryption(imgChange,buf);
    imwrite("/home/xiaozhou/Documents/MATLAB/test/lena_result_change.bmp", imgChange);

}

void chaosPrePro() {

    int t = 1000;

    for (int i = 0; i < t; i++) {
        double temp_x = 1 - sine2D_a * sin(sine2D_x) * sin(sine2D_x) + sine2D_y;
        temp_x = temp_x - (int)temp_x;
        double temp_y = sine2D_b * sine2D_x;
        temp_y = temp_y - (int)temp_y;
        sine2D_x = temp_x;
        sine2D_y = temp_y;
    }
    for (int i = 0; i < t; i++) {
        logistic_x = logistic_u * logistic_x * (1 - logistic_x);
    }

}


void imgEncryption(Mat &img,unsigned char buf[]) {
    reSetKey();
    updateKeys(buf);
    chaosPrePro();
    int channels = img.channels();
    int rows = img.rows;
    int cols = img.cols * channels;
    Mat final = img.clone();
    //如果图像通道数大于1，则根据图像通道数进行分割
    if (channels > 1) {
        Mat* channel = new Mat[channels];
        split(img, channel);
        //将三个通道重新排列，保存到一个图像中，图像大小为row*3*col
        for (int i = 0; i < channels - 1; i++) {
            hconcat(channel[i], channel[i + 1], channel[i + 1]);
        }
        final = channel[channels - 1].clone();
        delete[] channel;
    }

    //利用2DHSM混沌序列置乱图像的行和列
    imgConfusion(final);

    /*
    chaos用于每一列的混沌序列选择
    update用于更新每一列使用的混沌序列的初始密钥
    */
    int* chaos = new int[cols + rows];
    double* update = new double[cols + rows];
    chaosSelectByCol(chaos, update, cols + rows);

    //cuda部分
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    //dna加密图像所用的序列
    unsigned char* dev_dnaProSequence;
    cudaStatus = cudaMalloc((void**)&dev_dnaProSequence, static_cast<long long>(rows)* cols * 10 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, " dev_dnaProSequence cudaMalloc failed!");
    }

    double* dev_update;
    cudaStatus = cudaMalloc((void**)&dev_update, (rows + cols) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_update cudaMalloc failed!\n");
    }
    cudaStatus = cudaMemcpy(dev_update, update, (rows + cols) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_update cudaMemcpy failed!\n");
    }


    int* dev_chaos;
    cudaStatus = cudaMalloc((void**)&dev_chaos, (rows + cols) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_chaos cudaMalloc failed!\n");
    }
    cudaStatus = cudaMemcpy(dev_chaos, chaos, (rows + cols) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_chaos cudaMemcpy failed!\n");
    }


    //获取每一列所使用的混沌序列
    getDnaProSequenceKernel<<<(cols+128-1)/128, 128>>> (dev_dnaProSequence, dev_update, dev_chaos, rows,cols);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "getDnaEncodeRulesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "getDnaEncodeRulesKernel cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    //dna编码
    uchar* dev_dna = 0;
    cudaStatus = cudaMalloc((void**)&dev_dna, static_cast<long long>(rows)* cols * 4 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_dna cudaMalloc failed!\n");
    }

    unsigned char* dev_img = 0;
    cudaStatus = cudaMalloc((void**)&dev_img, static_cast<long long>(rows)* cols * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_img cudaMalloc failed!\n");
    }
    cudaStatus = cudaMemcpy(dev_img, final.data, static_cast<long long>(rows)* cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_img cudaMemcpy failed!\n");
    }

    imgDnaEncodeKernel <<<dim3((cols+128-1)/128,rows), 128 >>> (dev_img, dev_dna, dev_dnaProSequence, rows, cols);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "imgDnaEncodeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching imgDnaEncodeKernel!\n", cudaStatus);
    }


    //DNA-S盒替换
    dna_sbox_replaceKernel << <dim3((cols + 128 - 1) / 128, rows * 2), 128 >> > (dev_dna, dev_dnaProSequence, rows, cols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dna_sbox_replaceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dna_sbox_replaceKernel!\n", cudaStatus);
    }


    //将随机序列编码成dna序列，用于dna异或操作
    unsigned char* dev_dnaXor;
    cudaStatus = cudaMalloc((void**)&dev_dnaXor, static_cast<long long>(rows)* cols * 4 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_dnaXor cudaMalloc failed!\n");
    }
    getDnaXorSequenceKernel <<<(cols+128-1)/128, 128 >>> (dev_dnaXor, dev_dnaProSequence, rows, cols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaXorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaXorKernel!\n", cudaStatus);
    }

    //将随机dna序列与明文图像的dna进行异或操作
    dnaXorKernel <<<(rows*cols*4+128-1)/128, 128 >>> (dev_dna, dev_dnaXor, rows*cols*4);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaXorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaXorKernel!\n", cudaStatus);
    }

    //生成加操作序列
    unsigned char* & dev_dnaAdd = dev_dnaXor;//指针变量的引用，这样不必在gpu额外分配空间，节省时间
    getDnaAddSequenceKernel <<<(cols+128-1)/128, 128 >>> (dev_dnaAdd, dev_dnaProSequence, rows, cols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaXorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaXorKernel!\n", cudaStatus);
    }

    dnaAddKernel <<<(rows*cols*4+128-1)/128, 128 >>> (dev_dna, dev_dnaAdd, rows*cols*4);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaXorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaXorKernel!\n", cudaStatus);
    }

    //dna解码
    dnaDecodeKernel <<<dim3((cols+128-1)/128,rows), 128 >>> (dev_dna, dev_img, dev_dnaProSequence, rows, cols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dnaDecodeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dnaDecodeKernel!\n", cudaStatus);
    }
    cudaStatus=cudaMemcpy(final.data, dev_img, sizeof(unsigned char) * static_cast<long long>(rows)* cols, cudaMemcpyDeviceToHost);
    if(cudaStatus!= cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus= cudaDeviceReset();
    if(cudaStatus!= cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    //imshow("final fianl", final); waitKey(0);
    if (channels > 1) {
        //分割图像然后通道合并
        Mat* channel=new Mat[3];
        channel[0]=Mat(final,Rect(0,0,img.cols,img.rows));
        channel[1]=Mat(final,Rect(img.cols,0,img.cols,img.rows));
        channel[2]=Mat(final,Rect(img.cols*2,0,img.cols,img.rows));

        merge(channel,3,img);

        delete[] channel;
    }
    else {
        img=final.clone();
    }

}

void reSetKey() {
//sine混沌映射初始值和参数值
    std_a=3.764864654565236;//0-2*pi
    std_b=1.598741258692525;//0-2*pi
    std_k=17.589465464565456;//>0

//三维猫映射初始值和参数值
    cat3D_x = 0.192417345678913;//0-1
    cat3D_y = 0.556712345678916;//0-1
    cat3D_z = 0.932112345678123;//0-1
//洛伦兹混沌映射初始值和参数值
    lorenz_y = 0.786545641346986;//0-1
    lorenz_z = 0.253456749812345;//0-1
    lorenz_q = 0.598745498583658;//0-1

//sine2D混沌映射初始值和参数值
    sine2D_a= 37.857334516296548;//范围--负无穷到正无穷
    sine2D_b= 17.346265259595841;//范围--负无穷到正无穷

    sine2D_x= 0.265916526436985;//0-1
    sine2D_y= 0.695234546969854;//0-1

    logistic_x = 0.387368565654668;//0-1
}
