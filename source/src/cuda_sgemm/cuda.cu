#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>


//unoptimized this is 10 times slower than calling cublaSgemm
//but I can't figure out what the deal is with sGemm...
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

void modify (float *A, int lda, float *B, int ldb, float *C, int ldc, float alpha,float beta, int M, int N, int K)
{
	dim3 grids(16,16);
	dim3 blocks(56, 1250);
	sgemm<<<blocks,grids>>>(A, lda, B, ldb, C, ldc,alpha,beta, M,N,K);

// can't get culbasSgemm to follow the data layout I used.  My bad...I guess
//  cublasInit();
//	cublasSgemm('N','N', M,N,K,
//			alpha,
//			A, lda,
//			B, ldb,
//			beta, C, ldc);
//	cublasStatus stat = cublasGetError();
//	if (stat != CUBLAS_STATUS_SUCCESS){
//		printf("Error # %d: sgemm failed\n", stat);
//	}
//  cublasShutdown();

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
    float* device_A, *device_B, *device_C;

    float* a = 0;
    a = (float *)malloc (ww.row * data.col * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    unsigned int timer;
    cutCreateTimer(&timer);
    double time,total_time;

    total_time = 0;
    cutResetTimer(timer);
    cutStartTimer(timer);

//    for (int i=0; i<ww.row; i++){
//    	for (int j=0; j<data.col; j++){
//    		a[i * data.col + j] = 0;
//    		for (int k=0; k<16; k++){
//    			a[i * data.col + j] += ww.data[i * ww.col + k] * data.data[k * data.col + j];
//    		}
//    	}
//	}
//	for (int i=0; i<16; i++){
//    	printf("%f ", a[i]);
//    }
//	printf("\n");

//    for (int i=0; i<ww.row; i++){
//		for (int j=0; j<data.col; j++){
//			a[i * data.col + j] = 0;
//			for (int k=0; k<16; k++){
//				a[i * data.col + j] += ww.data[i * ww.col + k] * data.data[k * data.col + j];
//			}
//		}
//	}
//	for (int i=0; i<16; i++){
//		printf("%f ", a[i]);
//	}
//	printf("\n");
	float sum = 0;
	for (int i=0; i<16; i++){
		sum += ww.data[896*i] * data.data[ i];
//		printf("%f ", ww.data[i] * data.data[i]);
	}
	printf("%f \n",sum);
//	for (int i=0; i<16; i++){
//		printf("%f ", data.data[20000 * i]);
//	}
//	printf("\n");

    printf("setup matrix A %d %d\n", ww.row, ww.col);
    cutilSafeCall(cudaMalloc((void**)&device_A, sizeof(float) * ww.row * ww.col));
    cutilSafeCall(cudaMemcpy(device_A, ww.data, sizeof(float) * ww.row * ww.col, cudaMemcpyHostToDevice));

    printf("setup matrix B %d %d\n", data.row, data.col);
    cutilSafeCall(cudaMalloc((void**)&device_B, sizeof(float) * data.row*data.col));
    cutilSafeCall(cudaMemcpy(device_B, data.data, sizeof(float) * data.row * data.col, cudaMemcpyHostToDevice));
    //setupMatrix(device_B, a, 1, data.row, data.col);

    printf("setup matrix C %d %d\n", ww.row, data.col);
    cutilSafeCall(cudaMalloc((void**)&device_C, sizeof(float) * data.col * ww.row));
    for (int i=0; i<ww.row; i++){
    	for (int j=0; j<data.col; j++){
    		a[j * ww.row + i] = ww2.data[j];
    	}
    }
    cudaMemcpy(device_C, a, sizeof(float) * ww.row * data.col, cudaMemcpyHostToDevice);

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;
    printf("Initialization time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);
    modify (device_A, ww.row, device_B, data.row, device_C, ww.row, 2.0, -1.0, ww.row, data.col, ww.col);
    cudaThreadSynchronize();

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Run time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);
    cutilSafeCall(cudaMemcpy(a, device_C, sizeof(float) * ww.row * data.col, cudaMemcpyDeviceToHost));

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Transfer back time %f\n\n", time);

    printf("Total Time: %f\n\n", total_time);
    cudaFree (device_A);
    cudaFree (device_B);
    cudaFree (device_C);

    int new_ww_count[896];
    for (int i=0; i< 896; i++){
    	new_ww_count[i] = 0;
    }
	int argmax = 0;
	float max_val = 0;
	for (int i=0; i<20000; i++){
		argmax = 0;
		max_val = -100000;
		for (int j=0; j<896; j++){
			if (max_val < a[j * 20000 + i]){
				argmax = j;
				max_val = a[j * 20000 + i];
			}
		}

		new_ww_count[argmax]++;
	}

	int counter = 0;
	for (int i=0; i<56; i++){
		for (int j=0; j<16; j++){
			printf("%d ", new_ww_count[i * 16 + j]);
			counter += new_ww_count[i * 16 + j];
		}
		printf("\n");
	}

    for (int i=890; i<900; i++)
    	printf("%f ", a[i]);

	delete a;
    return EXIT_SUCCESS;
}


