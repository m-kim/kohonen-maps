#include <iostream>

#ifndef MATRIX_H
#define MATRIX_H

#define ROW_MAJOR 0
#define COLUMN_MAJOR 1
extern "C" void sgesvd_(const char* jobu, const char* jobvt, const int* M, const int* N,
        float* A, const int* lda, float* S, float* U, const int* ldu,
        float* VT, const int* ldvt, float* work,const int* lwork, const
        int* info);
extern "C" void sgesdd_(char *jobz, int *m, int *n,
		float *a, int *lda, float *s, float *u, int *ldu,
		float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info);


template<class T>
class MATRIX
{
public:
	T *data;
	int row;
	int col;
	virtual T& operator()(int _row, int _col){
		return this->data[_row + this->row * _col];
	}
	virtual T operator() (unsigned _row, unsigned _col) const{
		return this->data[_row + this->row * _col];
	}
	virtual void print(){
		for (int i=0; i<this->row; i++){
			for (int j=0; j<this->col; j++){
				printf("%f ",(*this)(i,j));
			}
			printf("\n");
		}
	}
};


template<class TYPE, int ORDER = 0>
class ORDERED_MATRIX:public MATRIX<TYPE>
{
public:
	TYPE& operator()(int _row, int _col){
		if (ORDER == COLUMN_MAJOR)
			return this->data[_row + this->row * _col];
		else
			return this->data[_row * this->col + _col];
	}

	TYPE operator() (unsigned _row, unsigned _col) const{
		if (ORDER == COLUMN_MAJOR)
			return this->data[_row + this->row * _col];
		else
			return this->data[_row * this->col + _col];

	}
	TYPE * mean()
	{
		unsigned int _row = this->row;
		unsigned int _col = this->col;
		//get the mean value for each row...
		float *ret = (float*)malloc(sizeof(float) * this->col);
		for(int i=0; i<_col;i++){
			ret[i] = 0.0;
			for(int j=0; j<_row; j++){
				ret[i] += (*this)(j,i);
			}
			ret[i] = ret[i] / _row;
		}
		return ret;
	}

	TYPE * cov()
	{
		int _col = this->col;
		int _row = this->row;
		TYPE *covariance = (TYPE*)malloc(sizeof(TYPE) * this->col * this->col);
		TYPE *mean_val = mean();
		for(int i=0; i< _col; i++){
			for (int j=0; j< _col;j++){
					covariance[i * _col + j] = 0;
					for (int k=0; k < _row; k++){
							covariance[i * _col + j] += ((*this)(k,i) - mean_val[i]) * ((*this)(k,j) - mean_val[j]);
					}
					covariance[i * _col + j] /= (_row - 1);
#if DEBUG_PRINT
					printf("%f ", covariance[i * _col + j]);
#endif
			}
#if DEBUG_PRINT
			printf("\n");
#endif
		}
		delete mean_val;
		return covariance;
	}
	void normalize()
	{
		unsigned int _row = this->row;
		unsigned int _col = this->col;
		float sum = 0;

		for (int i=0;i<_row; i++){
			sum = 0;
			for (int j=0; j<_col; j++){
				(*this)(i,j) = fabs((*this)(i,j));
				sum += (*this)(i,j);

			}
			for (int j=0;j<_col; j++){
				(*this)(i,j) /= sum;
			}
		}
	}

	void pca(MATRIX<TYPE> pca1, MATRIX<TYPE> pca2)
	{
		printf("Entering PCA...\n");
		//16 x 20000 means a 16x16 covariance matrix
		float *cov_mat = cov();

		unsigned int _row = this->row;
		unsigned int _col = this->col;


		char JOBU = 'N';
		char JOBV = 'A';

		int svd_M = _col;
		int svd_N = _col;
		int lda = _col;
		int ldu = _col;
		int ldv = _col;
		int info;

		int lwork = 3 * svd_M * svd_M + 4 * svd_M * svd_M + 4 * svd_M;;
		float s[_col];
		float uu[_col*_col];
		float vv[_col*_col];
		float wk[lwork];

		sgesvd_(&JOBU, &JOBV,
				&svd_N, &svd_M,
				cov_mat, &lda,
				s,
				uu, &ldu,
				vv, &ldv,
				wk, &lwork,
				&info);

	//	int iwork[8 * svd_M];
	//	//allegedly faster than sgesvd, but more memory required
	//	//vector size: 3*m*m + 4*m*m + 4*m
	//	//http://www.netlib.org/lapack/double/dgesdd.f
	//	//iwork, dimension (8*min(M,N))
	//	sgesdd_(&JOBV,
	//			&svd_N, &svd_M,
	//			cov_mat, &lda,
	//			s,
	//			uu, &ldu,
	//			vv, &ldv,
	//			wk, &lwork,
	//			iwork,
	//			&info);


		for (int i=0; i<_col; i++){
			pca1.data[i] = vv[i * _col];
			pca2.data[i] = vv[i * _col + 1];
		}
		printf("PCA finished...\n");
	//	memcpy(pca1, &vv[0], sizeof(float)* mat_m);
	//	memcpy(pca2, &vv[1], sizeof(float)* mat_m);

	#if DEBUG_PRINT
		for (int i=0; i<x.col; i++){
			printf("\t%f ", s[i]);
			for (int j=0; j<x.col; j++)
				printf("%f ", vv[IDX2C(i,j, x.col)]);
			printf("\n");
		}
		printf("\n");
		printf("%d\n",info);
	#endif
		delete cov_mat;
	}
};
#endif
