
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "external_dependency.h"
#include <cblas.h>
#include <cstdlib>
#include <cstring>

extern "C" int runCudasGemm();

//void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
//                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
//                 const int K, const float alpha, const float *A,
//                 const int lda, const float *B, const int ldb,
//                 const float beta, float *C, const int ldc);


void setMatrix(float *mat, float in, int m)
{
	for (int i=0; i<m; i++){
		mat[i] = in;

	}
}
int main( int argc, char **argv )
{
//	int m = 11;
//	float *A = (float*)malloc(sizeof(float) * m * m);
//	float *B = (float*)malloc(sizeof(float) * m * m);
//	float *C = (float*)malloc(sizeof(float) * m * m);
//
//	float alpha = 1.0;
//	float beta = 0.0;
//
//
//	setMatrix(A, 1.0, sizeof(float) * m * m);
//	setMatrix(B, 1.0, sizeof(float) * m * m);
//	setMatrix(C, 1.0, sizeof(float) * m * m);
//
//	//CblasNoTrans = 'n'
//	//CblasTrans = 't'
//	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, alpha,
//			A, m,
//			B, m,
//			beta,
//			C, m);
//
//	for (int i=0; i<m; i++){
//		for (int j=0; j<m; j++){
//			printf("%f ", C[i * m + j]);
//
//		}
//		printf("\n");
//	}
//	//return doit();
	runCudasGemm();

};

