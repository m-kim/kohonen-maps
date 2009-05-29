
#include <stdio.h>
#include <cblas.h>

#include <cstdlib>
#include <cstring>
#include <time.h>

extern "C" void sgemm_( char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int * );

void setMatrix(float *mat, float in, int m)
{
	for (int i=0; i<m; i++){
		mat[i] = in;

	}
}
int main( int argc, char **argv )
{
	 int m = 0;
        if (argc > 1){
                m = atoi(argv[1]);
        }
        else{
                m = 4096;
        }

	time_t start;

	double dif;
	

	float *A = (float*)malloc(sizeof(float) * m * m);
	float *B = (float*)malloc(sizeof(float) * m * m);
	float *C = (float*)malloc(sizeof(float) * m * m);

	float alpha = 1.0;
	float beta = 0.0;

	start = clock();
	setMatrix(A, 1.0, m * m);
	setMatrix(B, 1.0, m * m);
	setMatrix(C, 1.0, m * m);

	//CblasNoTrans = 'n'
	//CblasTrans = 't'
	char transA = 'N';
	char transB = 'N';
	sgemm_(&transA, &transB, &m, &m, &m, &alpha,
			A, &m,
			B, &m,
			&beta,
			C, &m);


	dif = difftime(clock(),start)/(double)CLOCKS_PER_SEC;
	printf("%f secs\n", dif);

//	for (int i=0; i<m; i++){
//		for (int j=0; j<m; j++){
//			printf("%f ", C[i * m + j]);
//
//		}
//		printf("\n");
//	}
    return 0;
};

