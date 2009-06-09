#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>

//there's probably a much nicer way to do this...
//but lets try this for now
__global__ void coalesce(const float *ww, const float *data, float *ww2,
						int *ret,
						float alpha, float beta,
						int M, int N, int K)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	float sum = 0;
	float max_val = -10000;
	int argmax = 0;
	if (i < 20000){
		for (int j=0; j<M; j++){
			sum = 0;
			for (int k=0; k<K; k++){
				sum += ww[j * K + k] * data[k * N + i];
			}
			sum = alpha * sum + beta * ww2[j];
			//ww2[j * N + i] = sum;
			if (max_val < sum){
					argmax = j;
					max_val = sum;
			}
		}
		ret[argmax]++;
	}
}


//unoptimized this is 10 times slower than calling cublaSgemm
//but I can't figure out what the deal is with sGemm...
//row-major order...
__global__ void sgemm(const float *A, int lda, const float *B, int ldb, float *C, int ldc,
						float alpha, float beta,
						int M, int N, int K)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	float sum = 0;
	for (int k=0; k<K; k++){
		sum += A[i * K + k] * B[k * N + j];

	}

	C[i * N + j] = alpha * sum + beta * C[i * N + j];
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

extern "C" int runCudasGemm(MATRIX ww, MATRIX ww2, MATRIX data)
{
    float* device_A, *device_B, *device_ww2;
    float* a = 0;
    a = (float *)malloc (ww.row * data.col * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    int *device_ret = 0;
	int *ret = (int*)malloc(sizeof(int) * ww.row);

    unsigned int timer;
    cutCreateTimer(&timer);
    double time,total_time;

    total_time = 0;
    cutResetTimer(timer);
    cutStartTimer(timer);

    printf("setup matrix ww %d %d\n", ww.row, ww.col);
    cutilSafeCall(cudaMalloc((void**)&device_A, sizeof(float) * ww.row * ww.col));
    cutilSafeCall(cudaMemcpy(device_A, ww.data, sizeof(float) * ww.row * ww.col, cudaMemcpyHostToDevice));

    printf("setup matrix data %d %d\n", data.row, data.col);
    cutilSafeCall(cudaMalloc((void**)&device_B, sizeof(float) * data.row*data.col));
    cutilSafeCall(cudaMemcpy(device_B, data.data, sizeof(float) * data.row * data.col, cudaMemcpyHostToDevice));
    //setupMatrix(device_B, a, 1, data.row, data.col);

    printf("setup matrix ww2 %d %d\n", ww.row, data.col);
    cutilSafeCall(cudaMalloc((void**)&device_ww2, sizeof(float) * ww2.row * ww2.col));
    cutilSafeCall(cudaMemcpy(device_ww2, ww2.data, sizeof(float) * ww2.row * ww2.col, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMalloc((void**)&device_ret, sizeof(int) * ww.row));
    cutilSafeCall(cudaMemset((void*)device_ret, 0, sizeof(int) * ww.row));
    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;
    printf("Initialization time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);

    coalesce<<<80,256>>>(device_A, device_B, device_ww2, device_ret, 2.0, -1.0, ww.row,data.col, data.row);
    cudaThreadSynchronize();

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Run time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);
    cutilSafeCall(cudaMemcpy(ret, device_ret, sizeof(int) * ww.row , cudaMemcpyDeviceToHost));

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Transfer back time %f\n\n", time);

    printf("Total Time: %f\n\n", total_time);
    cudaFree (device_A);
    cudaFree (device_B);
    cudaFree (device_ww2);
	cudaFree(device_ret);

	int counter = 0;
	for (int i=0; i<56; i++){
		 for (int j=0; j<16; j++){
				printf("%d ", ret[i * 16 + j]);
				counter += ret[i * 16 + j];
		 }
		 printf("\n");
	}
	printf("%d\n",counter);


	delete a, ret;
    return EXIT_SUCCESS;
}



