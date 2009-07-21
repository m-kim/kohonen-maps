#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>
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
	dev_calc_ww2<<<IMAGE_MxN/128,128>>>(ww,ww2);
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
	int index = row + IMAGE_M * col;
	if (col < IMAGE_M){
		int imin = max(row - _beta, 0);
		int imax = min(row + _beta + 1, IMAGE_N);

		for (int x=imin; x<imax; x++){
			for (int k=0; k<device_vector_size[0]; k++){
				b[k + device_vector_size[0] * index] += a[k + device_vector_size[0] * (x + IMAGE_M * col)];
			}
			count[index] += ww_count[x + IMAGE_M * col];
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
	dim3 grid(IMAGE_M/16,IMAGE_N/16);
	dev_prepSum<<<grid,block>>>(a, b, ww_count, count, _beta);
	cudaThreadSynchronize();
}

__global__ void dev_prepSum2(float *a, float *b, uint *ww_count, uint *count, int _beta)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	int index = row + IMAGE_M * col;

	//	__shared__ float s_ww[IMAGE_N * IMAGE_M];

	if (col < IMAGE_M){
		int imin = max(col - _beta,0);
		int imax = min(col + _beta + 1,IMAGE_N);
		for (int x=imin; x<imax; x++){
			for (int k=0; k<device_vector_size[0]; k++){
				a[k + device_vector_size[0] * index ] += b[k + device_vector_size[0] * ( row + IMAGE_M * x) ];
			}
			ww_count[index] += count[row + IMAGE_M * x];
		}
	}
}
extern "C" void prepSum2(float *a, float *b, uint *ww_count, uint *count, int _beta)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_M/16,IMAGE_N/16);
	dev_prepSum2<<<grid,block>>>(a, b, ww_count, count, _beta);
	cudaThreadSynchronize();
}

__global__ void dev_updateWeights(float *ww, float *avg_weight, float alpha)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	int index =  device_vector_size[0] * (row + IMAGE_M * col);
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
	dim3 grid(IMAGE_M/16,IMAGE_N/16);
	dev_updateWeights<<<grid,block>>>(ww, avg_weight, alpha);
	cudaThreadSynchronize();
}

__global__ void dev_normalizeSum(float *a, unsigned int* ww_count)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	for (int k=0; k<device_vector_size[0]; k++){
		//cc_sum(k, i + IMAGE_M * j) = argh(k, i + IMAGE_M * j)/count(j,i);

		if (ww_count[row + IMAGE_M * col] == 0)
			a[k + device_vector_size[0] * ( row + IMAGE_M * col)] = 0;
		else
			a[k + device_vector_size[0] * ( row + IMAGE_M * col)] = a[k + device_vector_size[0] * (row + IMAGE_M * col)]/(float)ww_count[row + IMAGE_M * col];
	}
}
extern "C" void normalizeSum(float *a, unsigned int* ww_count)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_M/16,IMAGE_N/16);
	dev_normalizeSum<<<grid,block>>>(a, ww_count);
	cudaThreadSynchronize();
}

//Calculate argmax and sum the data vectors
__global__ void dev_reduce(uint *ret, uint *indices, float *ww_sum, const float *vec, const float *data, unsigned int *argmax, int index)
{
	int size = IMAGE_MxN;
	//using shared memory here will limit me...
	//initialize with hard coded numbers because compile error on variable initialization
//	__shared__ int argmax[1024];
//	__shared__ float s_vec[1024];

	int blocksize = REDUCE_BLOCKSIZE;
	int coalesce_num = size/blocksize;

	for (int i=0; i<IMAGE_MxN/REDUCE_BLOCKSIZE; i++){
		argmax[threadIdx.x + i * blocksize] = threadIdx.x + i * blocksize;
//		s_vec[threadIdx.x + i * blocksize] = vec[threadIdx.x + i * blocksize];
	}


	// Large number ->32
	for (int j=1; j < coalesce_num; j++){
		if (threadIdx.x + blocksize * j < IMAGE_MxN){
			argmax[threadIdx.x] = (vec[argmax[threadIdx.x]] > vec[argmax[j * blocksize + threadIdx.x]])?
						argmax[threadIdx.x]:argmax[j * blocksize + threadIdx.x];
		}
	}

	//32->16, 16->8, 8->4, 4->2, 2->1
	for (int i=0; i<LOG2_REDUCE_BLOCKSIZE; i++){
		__syncthreads();
		blocksize = blocksize/2;

		if (threadIdx.x < blocksize){
			argmax[threadIdx.x] = vec[ argmax[blocksize +threadIdx.x]] < vec[argmax[threadIdx.x]]? argmax[threadIdx.x]:(argmax[blocksize+threadIdx.x]);
		}
	}
	__syncthreads();
	if (threadIdx.x < 1){
		ret[ argmax[0] ]++;
		indices[index] = argmax[0];
	}
	//take the vector from data and save it to ww_sum
	if (threadIdx.x < device_vector_size[0])
		ww_sum[ argmax[0] *device_vector_size[0] + threadIdx.x] += data[index * device_vector_size[0] + threadIdx.x];
}

extern "C" void reduce(uint *ret, uint *indices, float *ww_sum, const float *vec, const float *data, unsigned int *argmax, int index)
{
	dev_reduce<<<1,REDUCE_BLOCKSIZE>>>(ret,indices,ww_sum, vec,data, argmax, index);
}

__global__ void dev_buildImage(uint *im, uint *labels, uint *indices)
{
	uint row = threadIdx.x + blockDim.x * blockIdx.x;
	uint col = threadIdx.y + blockDim.y * blockIdx.y;
	uint index = row + IMAGE_M * col;

	im[index] = LABEL_COUNT + 2;
	for (int i=0; i<device_data_size[0]; i++){
		if (indices[i] == index)
			im[index] = labels[i];
	}
}

extern "C" void buildImage(uint *im, uint *labels, uint *indices)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_M/16,IMAGE_N/16);

    dev_buildImage<<<grid, block>>>(im,labels, indices);
    cudaThreadSynchronize();
}

__global__ void buildSplitImage(uint *im, uint *labels, uint *indices, int g_index)
{
	uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
	uint tidy = threadIdx.y + blockDim.y * blockIdx.y;
	uint index = tidx * IMAGE_N + tidy;

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
			im[(y * 16 + j) * 512 + x * 16 + i] = LUMINANCE_ADJUSTMENT * ret[y * IMAGE_M + x];
		}
	}
}

extern "C" void expandSplitImage(uint *im, const uint *ret)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_M/16,IMAGE_N/16);

	dev_expandSplitImage<<<grid,block>>>(im, ret);
}

__global__ void dev_increaseLuminance(unsigned char *im)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (im[y * 512 + x] > 0)
		im[y * 512 + x] += 1;
}

extern "C" void cuda_increaseLuminance(unsigned char *im)
{
	dim3 grid(32,32);
	dim3 block(16,16);
	dev_increaseLuminance<<<grid, block>>>(im);
}



__global__ void dev_decreaseLuminance(unsigned char *im)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (im[y * 512 + x] > 0)
		im[y * 512 + x] -= 1;
}

extern "C" void cuda_decreaseLuminance(unsigned char *im)
{
	dim3 grid(32,32);
	dim3 block(16,16);
	dev_increaseLuminance<<<grid, block>>>(im);
}

__global__ void dev_expandLogImage(unsigned char *im, const uint *ret)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<512/IMAGE_M; i++){
		for (int j=0; j<512/IMAGE_M; j++){
			im[(y * 512/IMAGE_M + j) * 512 + x * 512/IMAGE_M + i] = ret[y * IMAGE_M + x];
		}
	}
}

extern "C" void expandLogImage(unsigned char *im, const uint *ret)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_M/16,IMAGE_N/16);
	dev_expandLogImage<<<grid, block>>>(im,ret);
}
__global__ void dev_expandConstantImage(uint *im, const uint *ret)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<512/IMAGE_M; i++){
		for (int j=0; j<512/IMAGE_N; j++){
			im[(y * 512/IMAGE_M + j) * 512 + x * 512/IMAGE_M + i] = constant_color[ret[y * IMAGE_M + x]];
		}
	}
}
extern "C" void expandConstantImage(uint *im, const uint *ret)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_M/16,IMAGE_N/16);

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
		for (int i=0; i<IMAGE_MxN; i++){
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
