#include "shared.h"
#include <cstdlib>
#include <clapack.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>

extern "C" int runCudasGemm(MATRIX ww, MATRIX data);
extern "C" void sgesvd_(const char* jobu, const char* jobvt, const int* M, const int* N,
        float* A, const int* lda, float* S, float* U, const int* ldu,
        float* VT, const int* ldvt, float* work,const int* lwork, const
        int* info);

int M ;
int N ;

int expansion = 4;

float bin1, bin2;
float stdDev(const MATRIX &mat)
{
	float mean_x = 0;
	for (int i=0; i<mat.row; i++){
		mean_x += mat.data[i];
	}
	mean_x /= mat.row;
	float sum = 0;
	for (int i=0; i<mat.row; i++){
		sum += pow(mat.data[i] - mean_x, 2);
	}
	return sqrt(sum / mat.row);
}

float dot(MATRIX one, float *two)
{
	float sum = 0;
	for (int i=0; i<one.row; i++){
		sum += one.data[i] * two[i];
	}
	return sum;
}

float * mean(const MATRIX dd)
{
	//get the mean value for each row...
	float *ret = (float*)malloc(sizeof(float) * dd.row);

	for(int i=0; i<dd.row;i++){
		ret[i] = 0.0;
		for(int j=0; j<dd.col; j++){
			ret[i] += dd.data[i * dd.col + j];
		}
		ret[i] = ret[i] / dd.col;
	}
	return ret;
}
void cov(const MATRIX dd, float *covariance)
{
	float *mean_val = mean(dd);
	for(int i=0; i<dd.row; i++){
		for (int j=0; j<dd.row;j++){
			covariance[i * dd.row + j] = 0;
			for (int k=0; k < dd.col; k++){
				covariance[i * dd.row + j] += (dd.data[i * dd.col + k] - mean_val[i]) * (dd.data[j * dd.col + k] - mean_val[j]);
			}
			covariance[i * dd.row + j] /= (dd.col -  1);
//#if DEBUG_PRINT
			printf("%f ", covariance[i * dd.row + j]);
//#endif
		}
//#if DEBUG_PRINT
		printf("\n");
//#endif
	}

	delete mean_val;
}

//column major order doesn't matter for sgesvd_
//the matrix pumped out of cov is singular
//however, the matrix returned will be in column major order
//so, that needs to be taken into account...
void pca(MATRIX x, MATRIX pca1, MATRIX pca2)
{
	//16 x 20000 means a 16x16 covariance matrix
	float *cov_mat = (float*)malloc(sizeof(float) * x.row * x.row);

	cov(x,cov_mat);
	char JOBU = 'A';
	char JOBV = 'A';

	int svd_M = x.row;
	int svd_N = x.row;
	int lda = x.row;
	int ldu = x.row;
	int ldv = x.row;
	int info;
	int lwork;
	float s[x.row];
	float uu[x.row*x.row];
	float vv[x.row*x.row];
	float wk[201];

	sgesvd_(&JOBU, &JOBV,
			&svd_N, &svd_M,
			cov_mat, &lda,
			s,
			uu, &ldu,
			vv, &ldv,
			wk, &lwork,
			&info);


	for (int i=0; i<x.row; i++){
		pca1.data[i] = vv[i];
		pca2.data[1] = vv[i + x.row];
	}
//	memcpy(pca1, &vv[0], sizeof(float)* mat_m);
//	memcpy(pca2, &vv[1], sizeof(float)* mat_m);

#if DEBUG_PRINT
	for (int i=0; i<x.row; i++){
		printf("\t%f ", s[i]);
		for (int j=0; j<x.row; j++)
			printf("%f ", vv[IDX2C(i,j, x.row)]);
		printf("\n");
	}
	printf("\n");
	printf("%d\n",info);
#endif
	delete cov_mat;
}

//normalizes along the column
//ie if its a 16 x 20000 matrix, it normalizes the 16 vector
static void normalize(MATRIX mat)
{
	float sum = 0;
	for (int i=0;i<mat.col; i++){
		sum = 0;
		for (int j=0; j<mat.row; j++){
			mat.data[i * mat.row + j] = fabs(mat.data[i * mat.row + j]);
			sum += mat.data[i * mat.row + j];
		}
		if (i < 1)
			printf("normalize %f \n", sum);
		for (int j=0;j<mat.row; j++){
			mat.data[i * mat.row + j] /= sum;
		}
	}
}

int make_data(int n,int S, int F,float weight, MATRIX pc1, MATRIX pc2, MATRIX x)
{

	float center_vec[F];
	for (int i=0; i<S; i++){
		for (int cv_f = 0; cv_f < F; cv_f++){
			center_vec[cv_f] = (float)rand() / (float)RAND_MAX - 0.5;
		}
		for (int j=0; j<n; j++){
			for (int cv_f = 0; cv_f < F; cv_f++){
				x.data[(i * n + j) * F + cv_f] = weight * center_vec[cv_f] + (float)rand()/ (float)RAND_MAX - 0.5;
				printf("%f ", x.data[(i * n + j) * F + cv_f]);
			}
			printf("\n");
		}
	}

}

int main( int argc, char **argv )
{
	if (argc > 1){
		M = atoi(argv[1]);
		N = atoi(argv[1]);
	}
	else{
		M = 28;
		N = 32;
	}
	MATRIX pc1;
	pc1.data = (float*)malloc(sizeof(float) * 16);
	pc1.row = 16;
	pc1.col = 1;

	MATRIX pc2;
	pc2.data = (float*)malloc(sizeof(float) * 16);
	pc2.row = 16;
	pc2.col = 1;

	MATRIX x;
	x.data = (float*)malloc(sizeof(float) * 20000*16);
	x.row = 16;
	x.col = 20000;

//	float dd[] = {.69, -1.31, .39, .09, 1.29,.49,.19,-.81,-.31,-.71
//				,.49, -1.21, .99, .29, 1.09, .79, -.31, -.81, -.31, -1.01
//				,1.00000,  -1.50000,  -0.39000,   0.00000,  -0.09000,   1.40000,   0.39000,  -0.90000,  -0.31000,  -0.41000
//				};
//	MATRIX mat;
//	mat.data= dd;
//	mat.row = 3;
//	mat.col = 10;
//	pca(mat, pc1, pc2);



//	make_data(1000, 20, 16, 3.0, pc1, pc2, x);
	std::ifstream file;
	file.open("tmp", std::ifstream::in);
	std::string str;
	memset(x.data, 0, sizeof(float) * 16 * 20000);
	int row = 0;
	int col = 0;
	while (file.good()){
		//getline will retrieve 20000 numbers...
		getline(file, str);
		char *tok = strtok((char*)str.c_str(), " ");
		while (tok != NULL){
			//16 rows by 20000 cols in the file
			x.data[col * x.row + row] = atof(tok);
			tok = strtok(NULL, " ");
			col++;
		}
		col = 0;
		row++;
	}
	file.close();

	normalize(x);





	//pca(x, pc1, pc2);

	float tmp[] ={-0.36220333, -0.35487528, -0.19582513,  0.01291018,  0.25750163,  0.15277131
			 ,-0.37524103  ,0.16958821,  0.14361155, -0.2686145,   0.33782259,  0.00555586
			  ,0.29207888,  0.27095362,  0.15164405, -0.2376786};
	memcpy(pc1.data, tmp, sizeof(float) * 16);

	float tmp2[] = {0.00260173,  0.02293971, -0.47482202,  0.09716536,  0.00760736,  0.30567733
	  ,0.04289583, -0.09414034,  0.18733382, -0.01131097, -0.39201071,  0.52470409
	  ,0.29333093, -0.26513936, -0.19181659, -0.05501617};
	memcpy(pc2.data, tmp2, sizeof(float) * 16);

	//mean0 == dm
	//dm should be shape = (16,)
	//dm is correct compared to python code...
	//it needed a reverse index
	float *dm = (float*)malloc(sizeof(float) * x.row);

	for(int i=0; i<x.row;i++){
		dm[i] = 0.0;
		for(int j=0; j<x.col; j++){
			dm[i] += x.data[j * x.row + i];
		}
		dm[i] = dm[i] / x.col;
	}

	//-----------------------------------------------------------------------------------------------------


	MATRIX pd1;
	pd1.data = (float*)malloc(sizeof(float) * 20000);
	pd1.row = 20000;
	pd1.col = 1;

	MATRIX pd2;
	pd2.data = (float*)malloc(sizeof(float) * 20000);
	pd2.row = 20000;
	pd2.col = 1;

	MATRIX data_dm;
	data_dm.data = (float*)malloc(sizeof(float) * 20000 * 16);
	data_dm.row = 16;
	data_dm.col = 20000;

	for (int i=0; i<data_dm.col; i++){
		for (int j=0; j<data_dm.row; j++){
			data_dm.data[i * data_dm.row + j] = x.data[i * x.row + j] - dm[j];
		}
		pd1.data[i] = dot(pc1, data_dm.data + i * 16);
		pd2.data[i] = dot(pc2, data_dm.data + i * 16);
	}
	printf("pd %f\n", pd1.data[0]);
	//scale map
	float std1 = stdDev(pd1);
	float std2 = stdDev(pd2);


	//resize...
	//ummm...this is only if M is "None"
	//M = int(N * std2 / std1);

	bin1 = 2 * expansion * std1 / N;
	bin2 = 2 * expansion * std2 / M;

//#if DEBUG_PRINT
	printf("Std dev: %f %f\n", std1,std2);
	printf("scale %f %f %d %d", bin1, bin2, M, N);
//#endif
	//init_ww
	float *b1 = (float*)malloc(sizeof(float) * 16);
	float *b2 = (float*)malloc(sizeof(float) * 16);
	for (int i=0; i<pc1.row; i++){
		b1[i] = pc1.data[i] * bin1;
		b2[i] = pc2.data[i] * bin2;
	}


	MATRIX ww;
	ww.data = (float*)malloc(sizeof(float) * M * N * 16);
	ww.row = 16;
	ww.col = N * M;

	for (int i=0; i<M; i++){
		for (int j=0; j<N; j++){
			for (int k=0; k<16; k++){
				ww.data[i * N + j] = dm[k] + b1[k] * (i - N/2) + b2[k]*(j-M/2);
			}
		}
	}

	//musical_chairs
	for (int i=0; i<16; i++){
		for (int j=0; j<M*N; j++){
			ww.data[i * M*N + j] = ww.data[i*M*N +j] * ww.data[i*M*N +j];
		}
	}
	//chunk
	//K = 20000
	//M = 16
	//N = 896 (32 * 28)
	//runCudasGemm(x,ww);

	delete pc1.data, pc2.data, x, dm, pd1, pd2, data_dm, b1,b2,ww;
};
