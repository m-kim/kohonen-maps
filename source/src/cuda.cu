#include <cublas.h>
#include <stdio.h>
#define M 11
#define N 11

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void modify (float *A, int lda, float *B, int ldb, float *C, int ldc, float alpha,float beta)
{
	cublasSgemm('N','N', M,M,M,
			alpha,
			A, lda,
			B, ldb,
			beta, C, ldc);
	cublasStatus stat = cublasGetError();
	if (stat != CUBLAS_STATUS_SUCCESS){
		printf("Error # %d: sgemm failed\n", stat);
	}
}

void setupMatrix(float *&device_matrix, float *host_mem, float set_num)
{
    cublasStatus stat;
    stat = cublasAlloc (M*N, sizeof(*host_mem), (void**)&device_matrix);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("device memory allocation failed");
        cublasShutdown();
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
        	host_mem[IDX2C(i,j,M)] = set_num;
        }
    }
    stat = cublasSetMatrix (M, N, sizeof(*host_mem), host_mem, M, device_matrix, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cublasFree (host_mem);
        cublasShutdown();
    }

}
extern "C" int runCudasGemm()
{
    cublasStatus stat;
    float* device_A, *device_B, *device_C;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    cublasInit();
    setupMatrix(device_A, a, 1);
    setupMatrix(device_B, a, 1);
    setupMatrix(device_C, a, 0);
    modify (device_A, M, device_B, M, device_C, M, 1.0, 0.0);

    for (int j = 0; j < N; j++) {
		for (int i = 0; i < M; i++) {
			a[IDX2C(i,j,M)] = 0;
		}
	}

    stat = cublasGetMatrix (M, N, sizeof(*a), device_C, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cublasFree (device_C);
        cublasShutdown();
        return EXIT_FAILURE;
    }
    cublasFree (device_A);
    cublasFree (device_B);
    cublasFree (device_C);

    cublasShutdown();
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }
    return EXIT_SUCCESS;
  }
