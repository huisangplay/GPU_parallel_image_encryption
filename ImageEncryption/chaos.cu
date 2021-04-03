#include"chaos.cuh"
void getSine2DSequence(double* X, double* Y, int N) {
    for (int i = 0; i < N; i++) {
        double temp_x = 1 - sine2D_a * sin(sine2D_x) * sin(sine2D_x) + sine2D_y;
        temp_x = temp_x - (int)temp_x;
        double temp_y = sine2D_b * sine2D_x;
        temp_y = temp_y - (int)temp_y;
        sine2D_x = temp_x;
        sine2D_y = temp_y;

        X[i] = sine2D_x;
        Y[i] = sine2D_y;
    }
}

void getLogisticSequence(double* X, int n)
{
    for (int i = 0; i < n; i++) {
        logistic_x = logistic_u * logistic_x * (1 - logistic_x);
        X[i] = logistic_x;
    }
}
