#include "shared.h"
#include <cstdlib>
#include <clapack.h>
#include <cstdio>
#include <cstring>

extern "C" int runCudasGemm(int M, int N);
extern "C" void sgesvd_(const char* jobu, const char* jobvt, const int* M, const int* N,
        float* A, const int* lda, float* S, float* U, const int* ldu,
        float* VT, const int* ldvt, float* work,const int* lwork, const
        int* info);


int M ;
int N ;

static void cov(const float *dd, int row, int col, float *cov)
{
	//get the mean value for each row...
	float mean[row];

	for(int i=0; i<row;i++){
		mean[i] = 0.0;
		for(int j=0; j<col; j++){
			mean[i] += dd[i * col + j];
		}
		mean[i] = mean[i] / row;
	}

	for(int i=0; i<row; i++){
		for (int j=0; j<row;j++){
			cov[i * row + j] = 0;
			for (int k=0; k < col; k++){
				//something is wrong here...this index isn't correct
				cov[i * row + j] += (dd[i * col + k] - mean[i]) * (dd[j * col + k] - mean[j]);
			}
			cov[i * row + j] /= (col -  1);
		}
	}
}


static void pca(float *x, float *pca1, float *pca2, int mat_m, int mat_n, int vector_size)
{
	//16 x 20000 means a 16x16 covariance matrix
	float *cov_mat = (float*)malloc(sizeof(float) * mat_m * mat_m);

	cov(x, mat_m, vector_size, cov_mat);
	char JOBU = 'A';
	char JOBV = 'A';

	int svd_M = mat_m;
	int svd_N = mat_n;
	int lda = mat_m;
	int ldu = mat_m;
	int ldv = mat_n;
	int info;
	int lwork;
	float s[mat_m];
	float uu[mat_m*mat_m];
	float vv[mat_m*mat_m];
	float wk[201];

	sgesvd_(&JOBU, &JOBV,
			&svd_N, &svd_M,
			cov_mat, &lda,
			s,
			uu, &ldu,
			vv, &ldv,
			wk, &lwork,
			&info);


	memcpy(pca1, &vv[0], sizeof(float)* mat_m);
	memcpy(pca2, &vv[1], sizeof(float)* mat_m);
//	for (int i=0; i<16; i++){
//		printf("%f ", pca1[i]);
//	}
//	printf("\n");
//	printf("%d\n",info);
	delete cov_mat;
}

static void normalize(float *x)
{
	float sum = 0;
	for (int i=0;i<N; i++){
		sum += x[i];
	}
	for (int i=0;i<N; i++){
		x[i] /= sum;
	}
}

int make_data(int n,int S, int F,float weight, float *pc1, float *pc2, float *x)
{

	float center_vec[F];
	for (int i=0; i<S; i++){
		for (int cv_f = 0; cv_f < F; cv_f++){
			center_vec[cv_f] = (float)rand() / RAND_MAX - 0.5;
		}

		for (int j=0; j<n; j++){
			for (int cv_f = 0; cv_f < F; cv_f++){
				x[(i * n + j) * F + cv_f] = weight * center_vec[cv_f] + (float)rand()/RAND_MAX - 0.5;
			}
		}
	}

	normalize(x);
	//run pca....
	//pca(x, pc1, pc2);
}

int main( int argc, char **argv )
{
	if (argc > 1){
		M = atoi(argv[1]);
		N = atoi(argv[1]);
	}
	else{
		M = 2048;
		N = 2048;
	}
	float *pc1 = (float*)malloc(sizeof(float) * 16);
	float *pc2 = (float*)malloc(sizeof(float) * 16);
	float *x = (float*)malloc(sizeof(float) * 20000*16);
	//make_data(1000, 20, 16, 30, pc1, pc2, x);

	float dd[] = {.69, -1.31, .39, .09, 1.29,.49,.19,-.81,-.31,-.71,
				.49, -1.21, .99, .29, 1.09, .79, -.31, -.81, -.31, -1.01};


	float cov_mat[4];
	 cov(dd, 2, 10, cov_mat);
	 for (int i=0; i<4;i++)
		 printf("%f ", cov_mat[i]);
	 printf("\n");
	//runCudasGemm(M,N);

	delete pc1, pc2, x;
};
