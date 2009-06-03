#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>



void modify (float *A, int lda, float *B, int ldb, float *C, int ldc, float alpha,float beta,int K, int M, int N)
{
	cublasSgemm('N','N', K,M,N,
			alpha,
			A, lda,
			B, ldb,
			beta, C, ldc);
	cublasStatus stat = cublasGetError();
	if (stat != CUBLAS_STATUS_SUCCESS){
		printf("Error # %d: sgemm failed\n", stat);
	}
}

void setupMatrix(float *&device_matrix, float *host_mem, float set_num, int M, int N)
{
    cublasStatus stat;
    stat = cublasAlloc (M*N, sizeof(*host_mem), (void**)&device_matrix);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("device memory allocation failed\n");
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
        	host_mem[IDX2C(i,j,M)] = set_num;//j * M + i + 1;
        }
    }
    stat = cublasSetMatrix (M, N, sizeof(*host_mem), host_mem, M, device_matrix, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed\n");
    }

}

extern "C" int runCudasGemm(MATRIX ww, MATRIX data)
{
    cublasStatus stat;
    float* device_A, *device_B, *device_C;
//    float* a = 0;
//    a = (float *)malloc (M * N * sizeof (*a));
//    if (!a) {
//        printf ("host memory allocation failed");
//        return EXIT_FAILURE;
//    }

    unsigned int timer;
    cutCreateTimer(&timer);
    double time,total_time;

    total_time = 0;
    cutResetTimer(timer);
    cutStartTimer(timer);
    cublasInit();

    //this isn't any faster...surprising...
//    cudaMalloc((void**)&device_A, sizeof(float) * M * N * 3);
//    device_B = device_A + M * N;
//    device_C = device_B + M * N;
    printf("setup matrix A\n");
    cutilSafeCall(cudaMalloc((void**)&device_A, sizeof(float) * data.row*data.col));
    cutilSafeCall(cudaMemcpy(device_A, data.data, sizeof(float) * data.row * data.col, cudaMemcpyHostToDevice));
    //setupMatrix(device_A, a, 1, M, N);
    printf("setup matrix B\n");
    cutilSafeCall(cudaMalloc((void**)&device_B, sizeof(float) * ww.row * ww.col));
    cutilSafeCall(cudaMemcpy(device_B, ww.data, sizeof(float) * ww.row * ww.col, cudaMemcpyHostToDevice));
    //setupMatrix(device_B, a, 1, M, N);
    printf("setup matrix C\n");
    cutilSafeCall(cudaMalloc((void**)&device_C, sizeof(float) * data.col * ww.col));
    //cudaMemcpy(device_A, ww, sizeof(float) * N * K, cudaMemcpyHostToDevice);
    //setupMatrix(device_C, a, 0, M, N);
    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;
    printf("Initialization time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);
    modify (device_A, data.col, device_B, ww.col, device_C, ww.col, 1.0, 0.0,data.row, data.col, ww.col);
    cudaThreadSynchronize();

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Run time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);
    //stat = cublasGetMatrix (data.row, ww.col, sizeof(float), device_C, data.row, data.data, data.row);
    cutilSafeCall(cudaMemcpy(data.data, device_C, sizeof(float) * data.row * ww.row, cudaMemcpyDeviceToHost));

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Transfer back time %f\n\n", time);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cublasFree (device_C);
        cublasShutdown();
        return EXIT_FAILURE;
    }

    printf("Total Time: %f\n\n", total_time);
    cublasFree (device_A);
    cublasFree (device_B);
    cublasFree (device_C);

    cublasShutdown();

    for (int j = 0; j < data.row; j+=512) {
        for (int i = 0; i < ww.col; i+=512) {
            //printf ("%7.0f ", a[IDX2C(i,j,M)]);
        }
        //printf ("\n");
    }
    return EXIT_SUCCESS;
}

