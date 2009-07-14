#include <iostream>
#include <cutil_inline.h>

#ifndef MATRIX_H
#define MATRIX_H

#define ROW_MAJOR 0
#define COLUMN_MAJOR 1
#define DEVICE 0
#define HOST 1

extern "C" void sgesvd_(const char* jobu, const char* jobvt, const int* M, const int* N,
        float* A, const int* lda, float* S, float* U, const int* ldu,
        float* VT, const int* ldvt, float* work,const int* lwork, const
        int* info);
extern "C" void sgesdd_(char *jobz, int *m, int *n,
		float *a, int *lda, float *s, float *u, int *ldu,
		float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info);


template<class T, int COMPUTE_TYPE>
class MATRIX
{
public:
	T *data;
	int row;
	int col;
	float toler;
	MATRIX<T,COMPUTE_TYPE>(){
		toler = .5;
	}
	MATRIX<T,COMPUTE_TYPE>(int _row, int _col){
		row = _row;
		col = _col;
		data = new T[row * col];

		toler = .5;
	}

	~MATRIX<T,COMPUTE_TYPE>(){
		if (COMPUTE_TYPE == DEVICE)
			cutilSafeCall(cudaFree( this->data ));
		else
			delete this->data;

		};
	virtual T& operator()(int _row, int _col){
		return this->data[_row + this->row * _col];
	}
	virtual T operator() (unsigned _row, unsigned _col) const{
		return this->data[_row + this->row * _col];
	}
	virtual void print(){
		for (int i=0; i<this->row; i++){
			for (int j=0; j<this->col; j++){

				std::cout << (*this)(i,j) << " ";
			}
			std::cout << std::endl;
		}
	}

	float stdDev()
	{
		float mean_x = 0;
		for (int i=0; i<row; i++){
			mean_x += data[i];
		}
		mean_x /= row;
		float sum = 0;
		for (int i=0; i<row; i++){
			sum += pow(data[i] - mean_x, 2);
		}
		return sqrt(sum / (row - 1));
	}

	void printSize(){
		std::cout << "[" << row << "," << col << "]" << std::endl;
	}
	float dot(MATRIX<T,COMPUTE_TYPE> &two, int col)
	{
		float sum = 0;
		for (int i=0; i<row; i++){
			sum += data[i] * two(col,i);//.data[i * two.col + col ];
		}
		return sum;
	}

};


template<class TYPE, int COMPUTE_TYPE, int ORDER = 0>
class ORDERED_MATRIX:public MATRIX<TYPE,COMPUTE_TYPE>
{
public:
	ORDERED_MATRIX<TYPE, COMPUTE_TYPE, ORDER>():MATRIX<TYPE, COMPUTE_TYPE>(){};
	ORDERED_MATRIX<TYPE, COMPUTE_TYPE, ORDER>(int _row, int _col):MATRIX<TYPE, COMPUTE_TYPE>(_row,_col){};


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
			}
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
	void pca(MATRIX<TYPE, COMPUTE_TYPE> &pca1, MATRIX<TYPE, COMPUTE_TYPE> &pca2)
	{
		svd(pca1, pca2);

//		ORDERED_MATRIX<float, COMPUTE_TYPE, COLUMN_MAJOR> cov_mat;
//		cov_mat.row = this->col;
//		cov_mat.col = this->col;
//		cov_mat.data = cov();
//		printf("cov done\n");
//
//		float n = power_method(cov_mat,pca1);
//		printf("finished\n");
//		deflation_method(cov_mat,pca1, pca2,n);
	}

	void svd(MATRIX<TYPE, COMPUTE_TYPE> &pca1, MATRIX<TYPE, COMPUTE_TYPE> &pca2)
	{

		float *cov_mat = cov();
		for (int i = 0; i<16; i++){
			for (int j=0; j<16; j++){
				printf("%f ", cov_mat[i * 16 + j]);
			}
			printf("\n");
		}
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
	//	memcpy(pca1, &vv[0], sizeof(float)* mat_m);
	//	memcpy(pca2, &vv[1], sizeof(float)* mat_m);

		delete cov_mat;
	}

	float power_method(ORDERED_MATRIX<float, COMPUTE_TYPE, COLUMN_MAJOR> &mat ,MATRIX<TYPE, COMPUTE_TYPE> &x)
	{
//		dd=1;
//		x=start;
//		n=10;
//		while dd> toler
//		pause
//		end
//		vec=x;
//		value=n;


		for (int i=0; i<mat.row; i++){
			x.data[i] = 1;
		}
		float y[mat.col];
		for (int i=0; i<mat.col; i++){
			y[i] = 0;
		}

		float dd = 1;
		float n = 10;
		int iter = 0;
		while (dd > this->toler){
			iter++;
			memset(y, 0, sizeof(float) * mat.col);
			//		y=A*x
			for (int i=0; i<mat.row; i++){
				for (int j=0; j<mat.col; j++){
					y[i] += mat(i,j) * x.data[j];
				}
			}
			float tmp = 1;
			for (int i=0; i<mat.row; i++){
				tmp += x.data[i] * x.data[i];
			}

			//		dd=abs(norm(x)-n);
			tmp = sqrt(tmp);
			dd = fabs(tmp - n);
			//		n=norm(x)
			n = tmp;
			//		x=y/n
			for (int i=0; i<mat.row; i++){
				x.data[i] = y[i] / n;
			}
		}
		float tmp = 1;
		for (int i=0; i<mat.col; i++){
			tmp += x.data[i] * x.data[i];
		}

		tmp = sqrt(tmp);
		for (int i=0; i<mat.col; i++){
			x.data[i] /= tmp;
		}
		return n;
	}

	void deflation_method(ORDERED_MATRIX<float, COMPUTE_TYPE, COLUMN_MAJOR> &mat, MATRIX<TYPE,COMPUTE_TYPE> &in,
							MATRIX<TYPE, COMPUTE_TYPE> &x, float n)
	{
		for (int i=0; i<mat.row; i++){
			for (int j=0; j<mat.col; j++){
				mat(i,j) -= n * in.data[i] * in.data[j];
			}
		}
		for (int i=0; i<mat.row; i++){
			x.data[i] = 1;
		}
		power_method(mat, x);
	}
};
#endif
