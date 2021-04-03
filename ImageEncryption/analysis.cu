#include"analysis.cuh"
void imgWriteTofileByBinary(Mat img) {
    std::ofstream outFile("my.dat", std::ios::out | std::ios::binary);
    for (int i = 0; i < img.rows * img.cols * img.channels(); i++) {
        outFile.write((char*)&img.data[i], sizeof(unsigned char));
    }
}

double getChiSquareTests(Mat img)
{
    double tvalue = static_cast<long double>(img.cols)* img.rows* img.channels() / 256.0;
    int* count = new int[256]();
    for (int i = 0; i < img.cols * img.rows * img.channels(); i++) {
        count[img.data[i]]++;
    }
    double result = 0;
    for (int i = 0; i < 256; i++) {
        result += (count[i] - tvalue) * (count[i] - tvalue) / tvalue;
    }
    return result;
}