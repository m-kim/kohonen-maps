#include <iostream>

#define ROW_MAJOR 0
#define COLUMN_MAJOR 1


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
		if (ORDER)
			return this->data[_row + this->row * _col];
		else
			return this->data[_row * this->col + _col];
	}

	TYPE operator() (unsigned _row, unsigned _col) const{
		if (ORDER)
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
//		if (ORDER == COLUMN_MAJOR){
			for(int i=0; i<_col;i++){
				ret[i] = 0.0;
				for(int j=0; j<_row; j++){
					ret[i] += (*this)(j,i);
				}
				ret[i] = ret[i] / this->row;
			}
//		}
//		else{
//		}
		return ret;
	}

	TYPE * cov()
	{
		int _col = this->col;
		int _row = this->row;
		TYPE *covariance = (TYPE*)malloc(sizeof(TYPE) * this->col * this->col);
		TYPE *mean_val = mean();
//		if (ORDER == COLUMN_MAJOR){
			for(int i=0; i< _col; i++){
				for (int j=0; j< _col;j++){
						covariance[i * _col + j] = 0;
						for (int k=0; k < _row; k++){
								covariance[i * _col + j] += ((*this)(k,i) - mean_val[i]) * ((*this)(k,j) - mean_val[j]);
						}
						covariance[i * _col + j] /= (_row - 1);
		//#if DEBUG_PRINT
						printf("%f ", covariance[i * _col + j]);
		//#endif
				}
		//#if DEBUG_PRINT
				printf("\n");
		//#endif
			}
//		}
		delete mean_val;
		return covariance;
	}
	void normalize()
	{
		unsigned int _row = this->row;
		unsigned int _col = this->col;
		float sum = 0;

//		if(ORDER == COLUMN_MAJOR){
			for (int i=0;i<_row; i++){
				sum = 0;
				for (int j=0; j<_col; j++){
					(*this)(i,j) = fabs((*this)(i,j));//.data[i + _row * j]);
					sum += this->data[i + _row * j];
					if (i<1)
						printf("sum %f\n",fabs(this->data[i + _row * j]));

				}
				for (int j=0;j<_col; j++){
					this->data[i + _row * j] /= sum;
				}
			}
//		}
	}

};
