#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>
#include <limits.h>

#define EPSILON 0.000001

float host_alpha[2];
int host_r = -1, host_beta[2];

__constant__ uint constant_color[COLOR_SIZE];
__constant__ uint device_vector_size[1];
__constant__ uint device_data_size[1];

extern "C" void setup(int VECTOR_SIZE, int DATA_SIZE)
{
    //setup color
	unsigned char color[COLOR_SIZE * 4];
//	for(unsigned int i=0; i<COLOR_SIZE * 4; i+=4){
//		color[i + 1] = (unsigned char)i;
//		color[i + 2] = (i + 64) % 256;
//		color[i + 3] = (i + 128) % 256;
//		color[i] = 0;
//	}


	memset(color, 0, COLOR_SIZE *4);
	//dark green
	color[0] = 29;
	color[1] = 75;
	color[2] = 41;
	color[3] = 0;

	//red
	color[4] = 255;
	color[5] = 0;
	color[6] = 0;
	color[7] = 0;

	//green
	color[8] = 0;
	color[9] = 255;
	color[10] = 0;
	color[11] = 0;

	//blue
	color[12] = 0;
	color[13] = 0;
	color[14] = 255;
	color[15] = 0;

	//yellow
	color[16] = 255;
	color[17] = 255;
	color[18] = 0;
	color[19] = 0;

	//purple
	color[20] = 255;
	color[21] = 0;
	color[22] = 255;
	color[23] = 0;

	//cyan
	color[24] = 0;
	color[25] = 255;
	color[26] = 255;
	color[27] = 0;

	//dark blue
	color[28] = 28;
	color[29] = 47;
	color[30] = 140;
	color[31] = 0;

	color[32] = 180;
	color[33] = 128;
	color[34] = 128;
	color[35] = 0;

		//this is the empty space
//	color[32] = 0;
//	color[33] = 0;
//	color[34] = 0;
//	color[35] = 0;


	cutilSafeCall(cudaMemcpyToSymbol(constant_color, color, sizeof(unsigned int) * COLOR_SIZE, 0));
	cutilSafeCall(cudaMemcpyToSymbol(device_vector_size, &VECTOR_SIZE, sizeof(unsigned int), 0));
	cutilSafeCall(cudaMemcpyToSymbol(device_data_size, &DATA_SIZE, sizeof(unsigned int), 0));
}
extern "C" void safeMemset(void *ptr, char value, unsigned int size)
{
	cutilSafeCall(cudaMemset(ptr, value, size));
}
__global__ void dev_calc_ww2(MATRIX_TYPE *ww, MATRIX_TYPE *ww2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j=0; j<device_vector_size[0]; j++){
		//this shouldn't be backwards...*sigh*
		ww2[i] += pow(ww[i * device_vector_size[0] + j ], 2);
	}
}


extern "C" void calc_ww2(MATRIX_TYPE *ww, MATRIX_TYPE *ww2)
{
	dev_calc_ww2<<<IMAGE_XxY/128,128>>>(ww,ww2);
    cudaThreadSynchronize();

}

__global__ void dev_cov(MATRIX_TYPE *data, MATRIX_TYPE *covariance, MATRIX_TYPE *mean_val)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	//TYPE *covariance = (TYPE*)malloc(sizeof(TYPE) * this->col * this->col);
	//TYPE *mean_val = mean();
	covariance[i * device_vector_size[0] + j] = 0;
	for (int k=0; k < device_data_size[0]; k++){
			covariance[i * device_vector_size[0] + j] += (data[k * device_vector_size[0] + i] - mean_val[i]) * (data[k * device_vector_size[0] + j] - mean_val[j]);
	}
	covariance[i * device_vector_size[0] + j] /= (device_data_size[0] - 1);
}

extern "C" void cov(MATRIX_TYPE *data, MATRIX_TYPE *covariance, MATRIX_TYPE *mean_val)
{
	dim3 block(16,16);
	dim3 grid(1,1);
	dev_cov<<<grid,block>>>(data,covariance,mean_val);
}
__global__ void dev_prepSum(float *a, float *b, uint *ww_count, uint *count, int _beta)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	int index = row + IMAGE_Y * col;
	if (col < IMAGE_X){
		int imin = max(row - _beta, 0);
		int imax = min(row + _beta + 1, IMAGE_Y);

		for (int x=imin; x<imax; x++){
			for (int k=0; k<device_vector_size[0]; k++){
				b[k + device_vector_size[0] * index] += a[k + device_vector_size[0] * (x + IMAGE_Y * col)];
			}
			count[index] += ww_count[x + IMAGE_Y * col];
		}
	}
}

__global__ void dev_mean(const MATRIX_TYPE *data, MATRIX_TYPE *ret)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	int j = threadIdx.y + blockDim.y * blockIdx.y;

	//get the mean value for each row...
	//float *ret = (float*)malloc(sizeof(float) * this->col);
//	for(int i=0; i<device_vector_size[0];i++){
		ret[i] = 0.0;
		for(int j=0; j<device_data_size[0]; j++){
			ret[i] += data[j * device_vector_size[0] + i];
		}
		ret[i] = ret[i] / device_data_size[0];
//	}

}

extern "C" void mean(MATRIX_TYPE *data, MATRIX_TYPE *ret)
{
	dev_mean<<<1, 16>>>(data, ret);
}
extern "C" void prepSum(float *a, float *b, uint *ww_count, uint *count, int _beta)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_Y/16,IMAGE_X/16);
	dev_prepSum<<<grid,block>>>(a, b, ww_count, count, _beta);
	cudaThreadSynchronize();
}

__global__ void dev_prepSum2(float *a, float *b, uint *ww_count, uint *count, int _beta)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	int index = row + IMAGE_Y * col;

	//	__shared__ float s_ww[IMAGE_N * IMAGE_M];

	if (col < IMAGE_X){
		int imin = max(col - _beta,0);
		int imax = min(col + _beta + 1,IMAGE_X);
		for (int x=imin; x<imax; x++){
			for (int k=0; k<device_vector_size[0]; k++){
				a[k + device_vector_size[0] * index ] += b[k + device_vector_size[0] * ( row + IMAGE_Y * x) ];
			}
			ww_count[index] += count[row + IMAGE_Y * x];
		}
	}
}
extern "C" void prepSum2(float *a, float *b, uint *ww_count, uint *count, int _beta)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_Y/16,IMAGE_X/16);
	dev_prepSum2<<<grid,block>>>(a, b, ww_count, count, _beta);
	cudaThreadSynchronize();
}

__global__ void dev_updateWeights(float *ww, float *avg_weight, float alpha)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	int index =  device_vector_size[0] * (row + IMAGE_Y * col);
	for (int i=0; i<device_vector_size[0]; i++){
		ww[i + index] = ww[i + index] + alpha * (avg_weight[i + index] - ww[i + index]);
	}

//	//we're using avg_weight as a cache
//	avg_weight[index] = 0.0;
//	for (int i=0; i<device_vector_size[0]; i++){
//		avg_weight[index] += ww[i + index];
//	}
//
//	for (int i=0; i<device_vector_size[0]; i++){
//		//instead of a check for zero, add some epsilon
//		if (abs(avg_weight[index]) < .00001)
//			ww[i + index] = 0;
//		else
//			ww[i + index] /= avg_weight[index];
//	}
}
extern "C" void cuda_updateWeights(float *ww, float *avg_weight, float alpha)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_Y/16,IMAGE_X/16);
	dev_updateWeights<<<grid,block>>>(ww, avg_weight, alpha);
	cudaThreadSynchronize();
}

__global__ void dev_normalizeSum(float *a, unsigned int* ww_count)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	for (int k=0; k<device_vector_size[0]; k++){
		//cc_sum(k, i + IMAGE_M * j) = argh(k, i + IMAGE_M * j)/count(j,i);

		if (ww_count[row + IMAGE_Y * col] == 0)
			a[k + device_vector_size[0] * ( row + IMAGE_Y * col)] = 0;
		else
			a[k + device_vector_size[0] * ( row + IMAGE_Y * col)] = a[k + device_vector_size[0] * (row + IMAGE_Y * col)]/(float)ww_count[row + IMAGE_Y * col];
	}
}
extern "C" void normalizeSum(float *a, unsigned int* ww_count)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_Y/16,IMAGE_X/16);
	dev_normalizeSum<<<grid,block>>>(a, ww_count);
	cudaThreadSynchronize();
}

__global__ void dev_reduce1(uint *ret, uint *indices, float *ww_sum, float *vec, const float *data, unsigned int *argmax, int index)
{
	__shared__ int s_argmax[256];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_argmax[tid] = argmax[i];
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s > 0; s >>=1) {
	   if (tid < s) {
			//sdata[tid] += sdata[tid + s];
		   if (vec[ s_argmax[tid]] < vec[ s_argmax[s + tid]]){
			   s_argmax[tid] = s_argmax[s+tid];
			   vec[ s_argmax[tid] ] = vec[ s_argmax[s + tid] ];

		   }
	   }
	   __syncthreads();
	}
//	if (tid < 32){
//		unsigned int s = 32;
//		if (vec[ s_argmax[tid] ] < vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   vec[ s_argmax[tid] ] = vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (vec[ s_argmax[tid] ] < vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   vec[ s_argmax[tid] ] = vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (vec[ s_argmax[tid] ] < vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   vec[ s_argmax[tid] ] = vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (vec[ s_argmax[tid] ] < vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   vec[ s_argmax[tid] ] = vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (vec[ s_argmax[tid] ] < vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   vec[ s_argmax[tid] ] = vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (vec[ s_argmax[tid] ] < vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   vec[ s_argmax[tid] ] = vec[ s_argmax[s + tid] ];
//		}
//	}

	if (tid == 0){
		ret[s_argmax[0]]++;
		indices[index] = s_argmax[0];
	}
	//take the vector from data and save it to ww_sum
	if (threadIdx.x < device_vector_size[0])
		ww_sum[ s_argmax[0] *device_vector_size[0] + threadIdx.x] += data[index * device_vector_size[0] + threadIdx.x];

}
//Calculate argmax and sum the data vectors
__global__ void dev_reduce(uint *ret, uint *indices, float *ww_sum, float *vec, const float *data, unsigned int *argmax, int index)
{
	__shared__ int s_argmax[256];
	__shared__ int s_vec[256];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_vec[tid] = vec[i];
	s_argmax[tid] = tid;
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s > 0; s >>=1) {

	   	if (tid < s){
	       	//sdata[tid] += sdata[tid + s];
		   if (s_vec[ s_argmax[tid] ] < s_vec[ s_argmax[s + tid] ]){
			   s_argmax[tid] = s_argmax[s+tid];
			   s_vec[ s_argmax[tid] ] = s_vec[ s_argmax[s + tid] ];
		   }
	    	//s_argmax[tid] = s_vec[ s_argmax[s + tid] ] < s_vec[ s_argmax[tid] ]? s_argmax[tid]:;
	   }
	   __syncthreads();
	}
//	if (tid < 32){
//		unsigned int s = 32;
//		if (s_vec[ s_argmax[tid] ] < s_vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   s_vec[ s_argmax[tid] ] = s_vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (s_vec[ s_argmax[tid] ] < s_vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   s_vec[ s_argmax[tid] ] = s_vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (s_vec[ s_argmax[tid] ] < s_vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   s_vec[ s_argmax[tid] ] = s_vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (s_vec[ s_argmax[tid] ] < s_vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   s_vec[ s_argmax[tid] ] = s_vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (s_vec[ s_argmax[tid] ] < s_vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   s_vec[ s_argmax[tid] ] = s_vec[ s_argmax[s + tid] ];
//		}
//		s /= 2;
//		if (s_vec[ s_argmax[tid] ] < s_vec[ s_argmax[s + tid] ]){
//		   s_argmax[tid] = s_argmax[s+tid];
//		   s_vec[ s_argmax[tid] ] = s_vec[ s_argmax[s + tid] ];
//		}
//	}
	if (tid == 0){
		argmax[blockIdx.x] = s_argmax[0] + blockDim.x * blockIdx.x;
		vec[blockIdx.x] = s_vec[0];
	}
}

extern "C" void reduce(uint *ret, uint *indices, float *ww_sum, float *vec, const float *data, unsigned int *argmax, int index)
{
	dev_reduce<<<IMAGE_X,IMAGE_Y>>>(ret,indices,ww_sum, vec,data, argmax, index);
	dev_reduce1<<<1,IMAGE_X>>>(ret,indices,ww_sum, vec, data,argmax,index);
}

__global__ void dev_buildImage(uint *im, uint *labels, uint *indices)
{
	uint row = threadIdx.x + blockDim.x * blockIdx.x;
	uint col = threadIdx.y + blockDim.y * blockIdx.y;
	uint index = row + IMAGE_Y * col;

	im[index] = LABEL_COUNT + 2;
	for (int i=0; i<device_data_size[0]; i++){
		if (indices[i] == index)
			im[index] = labels[i];
	}
}

extern "C" void buildImage(uint *im, uint *labels, uint *indices)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_Y/16,IMAGE_X/16);

    dev_buildImage<<<grid, block>>>(im,labels, indices);
    cudaThreadSynchronize();
}

__global__ void buildSplitImage(uint *im, uint *labels, uint *indices, int g_index)
{
	uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
	uint tidy = threadIdx.y + blockDim.y * blockIdx.y;
	uint index = tidx * IMAGE_Y + tidy;

	int genome[GENOMIC_DATA_COUNT];

	for (int i=0; i<GENOMIC_DATA_COUNT; i++)
		genome[i] = 0;

	for (int i=0; i<device_data_size[0]; i++){
		if (indices[i] == index){
			genome[ labels[i] ]++;
		}
	}

	int count = 0;
	for (int i=0; i<GENOMIC_DATA_COUNT; i++){
		count = 0;
		for (int j=0; j<GENOMIC_DATA_COUNT; j++){
			if (i != j)
				count += (genome[i] > genome[j]);
		}
		if (count == (GENOMIC_DATA_COUNT - 1)){
			im[index] = genome[g_index];
			return;
		}
	}
	im[index] = GENOMIC_DATA_COUNT;
}

__global__ void dev_expandSplitImage(uint *im, const uint *ret)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<16; i++){
		for (int j=0; j<16; j++){
			im[(y * 16 + j) * 512 + x * 16 + i] = LUMINANCE_ADJUSTMENT * ret[y * IMAGE_X + x];
		}
	}
}

extern "C" void expandSplitImage(uint *im, const uint *ret)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_Y/16,IMAGE_X/16);

	dev_expandSplitImage<<<grid,block>>>(im, ret);
}

__global__ void dev_increaseLuminance(unsigned int *im)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (im[y * 512 + x] > 0)
		im[y * 512 + x] += 1;
}

extern "C" void cuda_increaseLuminance(unsigned int *im)
{
	dim3 grid(32,32);
	dim3 block(16,16);
	dev_increaseLuminance<<<grid, block>>>(im);
}



__global__ void dev_decreaseLuminance(unsigned int *im)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (im[y * 512 + x] > 0)
		im[y * 512 + x] -= 1;
}

extern "C" void cuda_decreaseLuminance(unsigned int *im)
{
	dim3 grid(32,32);
	dim3 block(16,16);
	dev_increaseLuminance<<<grid, block>>>(im);
}

__global__ void dev_reduceLogImage(unsigned int *count, unsigned int *save)
{

	__shared__ int s_count[256];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_count[tid] = count[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s > 0; s >>=1) {
	   	if (tid < s){
	       	//sdata[tid] += sdata[tid + s];
		   if (s_count[ tid ] < s_count[ s + tid ]){
			   s_count[tid] = s_count[s+tid];

		   }
	    	//s_argmax[tid] = s_vec[ s_argmax[s + tid] ] < s_vec[ s_argmax[tid] ]? s_argmax[tid]:;
	   }
	   __syncthreads();
	}
	if (tid == 0){
		save[blockIdx.x] = s_count[0];
	}
}
__global__ void dev_reduceLogImage1(unsigned int *count)
{

	__shared__ int s_count[256];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_count[tid] = count[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s > 0; s >>=1) {
	   	if (tid < s){
	       	//sdata[tid] += sdata[tid + s];
		   if (s_count[ tid ] < s_count[ s + tid ]){
			   s_count[tid] = s_count[s+tid];

		   }
	    	//s_argmax[tid] = s_vec[ s_argmax[s + tid] ] < s_vec[ s_argmax[tid] ]? s_argmax[tid]:;
	   }
	   __syncthreads();
	}
	if (tid == 0){
		count[blockIdx.x] = s_count[0];
	}
}
__global__ void dev_expandLogImage(unsigned int *im, const uint *ret)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	unsigned int max = im[0];
	unsigned int scale = 0;
	if (max > 0)
		scale = INT_MAX/max;

	for (int i=0; i<512/IMAGE_Y; i++){
		for (int j=0; j<512/IMAGE_X; j++){
			im[(y * 512/IMAGE_X + j) * 512 + x * 512/IMAGE_Y + i] = scale * ret[y * IMAGE_Y + x];
		}
	}
}

extern "C" void expandLogImage(unsigned int *im, uint *ret)
{
	dev_reduceLogImage<<<IMAGE_X, IMAGE_Y>>>(ret, im);
	dev_reduceLogImage1<<<1, IMAGE_X>>>(im);

	//now the
	dim3 block(16,16);
	dim3 grid(IMAGE_Y/16,IMAGE_X/16);
	dev_expandLogImage<<<grid, block>>>(im,ret);
}

__global__ void dev_expandConstantImage(uint *im, const uint *ret)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<512/IMAGE_Y; i++){
		for (int j=0; j<512/IMAGE_X; j++){
			im[(col * 512/IMAGE_X + j) * 512 + row * 512/IMAGE_Y + i] = constant_color[ret[col * IMAGE_Y + row]];
		}
	}
}
extern "C" void expandConstantImage(uint *im, const uint *ret)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_Y/16,IMAGE_X/16);

	dev_expandConstantImage<<<grid,block>>>(im,ret);

}

__global__ void dev_findWeightVector(MATRIX_TYPE *weight, MATRIX_TYPE *data, unsigned int *indices)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < device_data_size[0]){
		//__shared__ MATRIX_TYPE s_data[IMAGE_MxN];

//		for (int i=0; i<device_vector_size[0]; i++){
//			s_data[i] = data[index * device_vector_size[0] + i];
//		}

		int argmax = 0;
		MATRIX_TYPE max = -1000000;
		for (int i=0; i<IMAGE_XxY; i++){
			MATRIX_TYPE sum = 0;

			for (int j=0; j<device_vector_size[0]; j++){
				sum += 2 * data[index * device_vector_size[0] + j] * weight[i * device_vector_size[0] + j] -
												weight[i * device_vector_size[0] + j] * weight[ i * device_vector_size[0] + j];
			}
			if (sum > max){
				max = sum;
				argmax = i;
			}
		}
		indices[index] = argmax;
	}
}

__global__ void dev_calcSum(MATRIX_TYPE *ww_sum, MATRIX_TYPE *data, unsigned int *indices)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i=0; i<device_vector_size[0]; i++){
		ww_sum[ indices[index] *device_vector_size[0] + i] += data[indices[index] * device_vector_size[0] + i];
	}
}



extern "C" void findWeightVector(MATRIX_TYPE *ww_sum, MATRIX_TYPE *weight, MATRIX_TYPE *data,unsigned int *indices)
{
	dev_findWeightVector<<<22436/32, 32>>>(weight, data, indices);

	cudaThreadSynchronize();

//	dev_calcSum<<<IMAGE_MxN/32, 32>>>(ww_sum, data, indices);
//	cudaThreadSynchronize();
}
