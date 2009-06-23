#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>
#define EPSILON 0.000001
#define REDUCE_BLOCKSIZE 256
#define LOG2_REDUCE_BLOCKSIZE 8

MATRIX<MATRIX_TYPE> device_ww, device_data, device_ww2, device_save, device_sum, device_scratch;
MATRIX<unsigned int> device_labels, device_indices, device_ww_count, device_ret,device_ww_count2;
float* a;
unsigned int *ret, *indices;

float host_alpha[2];
int host_r = -1, host_beta[2];

__constant__ uint constant_color[256];
__constant__ int beta[2];

__global__ void calc_ww2(const MATRIX<MATRIX_TYPE> ww, MATRIX_TYPE *ww2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j=0; j<VECTOR_SIZE; j++){
		//this shouldn't be backwards...*sigh*
		ww2[i] += pow(ww.data[j * ww.row + i],2);
	}

}
__global__ void update_weights(float *a, float *b, uint *ww_count, uint *count, int _beta)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int slab = threadIdx.y + blockDim.y * blockIdx.y;
	int index = j * IMAGE_M + slab;

	if (slab < IMAGE_M){
		int _min = max(j - _beta, 0);
		int _max = min(j + _beta + 1, IMAGE_N);

		for (int i=0; i<VECTOR_SIZE; i++){  //vector size...
			for (int k= _min; k<_max; k++){
				b[i * IMAGE_MxN + index]  += a[i * IMAGE_MxN + k * IMAGE_M + slab];
			}
		}
		for (int k= _min; k<_max; k++){
			count[index] += ww_count[k * IMAGE_M + slab];
		}
	}
}

__global__ void update_weights2(float *ww, float *a, float *b, uint *ww_count, uint *count, int _beta, float _alpha)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int slab = threadIdx.y + blockDim.y * blockIdx.y;
	int index = j * IMAGE_M + slab;
//	__shared__ float s_ww[IMAGE_N * IMAGE_M];
	int _min = max(slab - _beta, 0);
	int _max = min(slab + _beta + 1, IMAGE_M);

	if (slab < IMAGE_M){

		for (int i=0; i<VECTOR_SIZE; i++){  //vector size...
			for (int k= _min; k<_max; k++){
				a[i * IMAGE_MxN + index]  += b[i * IMAGE_MxN + j * IMAGE_M + k];
			}
		}
		for (int k= _min; k<_max; k++){
			ww_count[index] += count[j * IMAGE_M + k];
		}

		for (int i=0; i<VECTOR_SIZE; i++){
			if (ww_count[index] == 0)
				a[i * IMAGE_MxN + index] = 0;
			else
				a[ i * IMAGE_MxN + index] = a[ i * IMAGE_MxN + index] / (ww_count[index] + EPSILON);
        	ww[i * IMAGE_MxN + index] = abs(ww[i * IMAGE_MxN + index]  +_alpha * (a[i * IMAGE_MxN + index] - ww[i * IMAGE_MxN + index]));
		}
		for (int i=0; i<VECTOR_SIZE; i++){
	    	ww[index] += ww[i * IMAGE_MxN + index];
		}
	}
	__syncthreads();
	if (slab < IMAGE_M){
		for (int i=0; i<VECTOR_SIZE; i++){
			if (ww[index] > 0)
				ww[i * IMAGE_MxN + index] = ww[i * IMAGE_MxN + index] / (ww[index]);
			else
				ww[i * IMAGE_MxN + index] = 0;

		}
	}
}

//Calculate argmax and sum the data vectors
__global__ void reduce(uint *ret, uint *indices, float *ww_sum, const float *vec, const float *data, int index)
{
	int size = 1024;
	//using shared memory here will limit me...
	//initialize with hard coded numbers because compile error on variable initialization
	__shared__ uint argmax[1024];
	__shared__ float s_vec[1024];

	int blocksize = REDUCE_BLOCKSIZE;
	int coalesce_num = size/blocksize;
	argmax[threadIdx.x] = threadIdx.x;

	for (int i=0; i<1024/REDUCE_BLOCKSIZE; i++){
		argmax[threadIdx.x + i * blocksize] = threadIdx.x + i * blocksize;
		s_vec[threadIdx.x + i * blocksize] = vec[threadIdx.x + i * blocksize];
	}


	// 256->32
	for (int j=1; j < coalesce_num; j++){
		if (threadIdx.x + blocksize * j < IMAGE_MxN){
			argmax[threadIdx.x] = (s_vec[argmax[threadIdx.x]] > s_vec[argmax[j * blocksize + threadIdx.x]])?
						argmax[threadIdx.x]:argmax[j * blocksize+threadIdx.x];
		}
	}

	//32->16, 16->8, 8->4, 4->2, 2->1
	for (int i=0; i<LOG2_REDUCE_BLOCKSIZE; i++){
		__syncthreads();
		blocksize = blocksize/2;

		if (threadIdx.x < blocksize){
			argmax[threadIdx.x] = s_vec[ argmax[blocksize +threadIdx.x]] < s_vec[argmax[threadIdx.x]]? argmax[threadIdx.x]:(argmax[blocksize+threadIdx.x]);
		}
	}
	__syncthreads();
	ret[ argmax[0] ]++;
	indices[index] = argmax[0];

	//take the vector from data and save it to ww_sum
	if (threadIdx.x < VECTOR_SIZE)
		ww_sum[ argmax[0] + threadIdx.x * IMAGE_MxN] += data[index * VECTOR_SIZE + threadIdx.x];//ww_sum[ 410 * 16 + threadIdx.x] = data[threadIdx.x];
}

__global__ void buildImage(uint *im, uint *labels, uint *indices)
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ int nn[IMAGE_N];
	__shared__ int mm[IMAGE_N];
	nn[threadIdx.x] = indices[i] / IMAGE_M;
	mm[threadIdx.x] = indices[i] - IMAGE_M * nn[threadIdx.x];
	im[ nn[threadIdx.x] * IMAGE_M + mm[threadIdx.x]] = labels[i] + 1;
}

__global__ void expandImage(uint *im, const uint *ret)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<16; i++){
		for (int j=0; j<16; j++){
			im[(y * 16 + j) * 512 + x * 16 + i] = constant_color[ret[y * IMAGE_M + x]] * ret[y * IMAGE_M + x];
		}
	}
}

extern "C" void cleanup()
{
    cudaFree (device_ww.data);
    cudaFree (device_data.data);
    cudaFree (device_ww2.data);
	cudaFree(device_indices.data);
	cudaFree(device_labels.data);
	delete a, ret, indices;
}
extern "C" void setupCuda(MATRIX<MATRIX_TYPE> ww,  MATRIX<MATRIX_TYPE> data, uint *labels, unsigned int *device_pbo)
{
    //setup color
	unsigned char color[1024];
	for(unsigned int i=0; i<255; i+=4){
		color[i] = (unsigned char)i;
		color[i + 1] = (i + 64) % 256;
		color[i + 2] = (i + 128) % 256;
		color[i + 3] = (i + 192) % 256;

	}
	cutilSafeCall(cudaMemcpyToSymbol(constant_color, color, sizeof(unsigned int) * 256, cudaMemcpyHostToDevice));

	host_beta[0] = 8;
	host_beta[1] = 8;
	host_alpha[0] = .6;
	host_alpha[1] = .6;

	cutilSafeCall(cudaMemcpyToSymbol(beta, host_beta, sizeof(int) * 2, cudaMemcpyHostToDevice));

	cudaMemset(device_pbo, 128, sizeof(unsigned int) * 512 * 512);

	device_labels.row = data.col;
	device_labels.col = 1;
	cutilSafeCall(cudaMalloc((void**)&device_labels.data, sizeof(uint) * data.col));
	cutilSafeCall(cudaMemcpy(device_labels.data, labels, sizeof(uint) * data.col, cudaMemcpyHostToDevice));

	device_ww_count.row = ww.row;
	device_ww_count.col = ww.col;
	cutilSafeCall(cudaMalloc((void**)&device_ww_count.data, sizeof(unsigned int) * device_ww_count.row));
    cutilSafeCall(cudaMemset((void*)device_ww_count.data, 0, sizeof(unsigned int) * device_ww_count.row));

	device_ww_count2.row = ww.row;
	device_ww_count2.col = ww.col;
	cutilSafeCall(cudaMalloc((void**)&device_ww_count2.data, sizeof(unsigned int) * device_ww_count2.row));
    cutilSafeCall(cudaMemset((void*)device_ww_count2.data, 0, sizeof(unsigned int) * device_ww_count2.row));

    device_indices.row = ww.row;
    device_indices.col = 1;
	cutilSafeCall(cudaMalloc((void**)&device_ret.data, sizeof(unsigned int) * ww.row));
    cutilSafeCall(cudaMemset((void*)device_ret.data, 0, sizeof(unsigned int) * ww.row));

    device_indices.row = data.col;
    device_indices.col = 1;
    cutilSafeCall(cudaMalloc((void**)&device_indices.data, sizeof(unsigned int) * data.col));
    cutilSafeCall(cudaMemset(device_indices.data, 0, sizeof(unsigned int) * data.col));

    device_ww.row = ww.row;
    device_ww.col = ww.col;
    printf("setup matrix ww %d %d\n", ww.row, ww.col);
    cutilSafeCall(cudaMalloc((void**)&device_ww.data, sizeof(float) * ww.row * ww.col));
    cutilSafeCall(cudaMemcpy(device_ww.data, ww.data, sizeof(float) * ww.row * ww.col, cudaMemcpyHostToDevice));
    for (int i=0; i<ww.row; i++){
    	for (int j=0; j<ww.col; j++){
    		cutilSafeCall(cudaMemcpy(
    				device_ww.data  + (j * ww.row + i),
    				ww.data + (i * ww.col + j), sizeof(float), cudaMemcpyHostToDevice));
    	}
    }

	device_ww2.row = IMAGE_N;
	device_ww2.col = IMAGE_M;
    printf("setup matrix ww2 %d %d\n", device_ww2.row, device_ww2.col);
    cutilSafeCall(cudaMalloc((void**)&device_ww2.data, sizeof(float) * device_ww2.row * device_ww2.col));
	cutilSafeCall(cudaMemset(device_ww2.data, 0, sizeof(float) * device_ww2.row * device_ww2.col));


    device_sum.row = ww.row;
    device_sum.col = ww.col;
    printf("setup matrix sum %d %d\n", device_sum.row, device_sum.col);
    cutilSafeCall(cudaMalloc((void**)&device_sum.data, sizeof(float) * device_sum.row * device_sum.col));
    cutilSafeCall(cudaMemset(device_sum.data, 0, sizeof(float) * device_sum.row * device_sum.col ));


    printf("setup matrix scractch %d %d\n", IMAGE_MxN, VECTOR_SIZE);
    cutilSafeCall(cudaMalloc((void**)&device_scratch.data, sizeof(float) * IMAGE_MxN * VECTOR_SIZE));
    cutilSafeCall(cudaMemset(device_scratch.data, 0, sizeof(float) * IMAGE_MxN * VECTOR_SIZE));

    device_save.row = device_ww2.row;
    device_save.col = device_ww2.col;
    device_save.data = device_scratch.data;

    a = (float *)malloc (ww.row * data.col * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
    }

	ret = (unsigned int*)malloc(sizeof(unsigned int) * ww.row);
	indices = (uint*)malloc(sizeof(uint) * data.col);

    device_data.row = data.row;
    device_data.col = data.col;
    printf("setup matrix data %d %d\n", device_data.row, device_data.col);
    cutilSafeCall(cudaMalloc((void**)&device_data, sizeof(float) * device_data.row*device_data.col));
    cutilSafeCall(cudaMemcpy(device_data.data, data.data, sizeof(float) * device_data.row * device_data.col, cudaMemcpyHostToDevice));
//    for (int i=0; i<data.row; i++){
//    	for (int j=0; j<data.col; j++){
//    		cutilSafeCall(cudaMemcpy(
//    				device_data.data  + (j * data.row + i),
//    				data.data + (i * data.col + j), sizeof(float), cudaMemcpyHostToDevice));
//    	}
//    }
}

extern "C" int runCuda(unsigned int *device_pbo)
{
	unsigned int timer;
    cutCreateTimer(&timer);
    double time,total_time;

    dim3 block;
    dim3 grid;

    total_time = 0;
    cutResetTimer(timer);
    cutStartTimer(timer);
    cutilSafeCall(cudaMemset((void*)device_ww_count.data, 0, sizeof(unsigned int) * device_ww_count.row));
    cutilSafeCall(cudaMemset((void*)device_ww_count2.data, 0, sizeof(unsigned int) * device_ww_count.row));

    cudaMemset(device_ww2.data, 0, sizeof(float) * device_ww2.row * device_ww2.col);

    //this is related to IMAGE_MXN
    calc_ww2<<<IMAGE_MxN/128,128>>>(device_ww,device_ww2.data);
    cudaMemcpy(a,device_ww2.data,sizeof(int) * device_ww2.row * device_ww2.col, cudaMemcpyDeviceToHost);

#if DEBUG_PRINT
    for (int i=0; i<IMAGE_N; i++){
    	for (int j=0; j<IMAGE_M; j++){
    		printf("%f ", a[i * IMAGE_M + j]);
    	}
    	printf("\n");
    }
#endif

    cudaThreadSynchronize();
 	cutilSafeCall(cudaMemcpy(device_save.data, device_ww2.data, sizeof(float) * device_ww2.row * device_ww2.col, cudaMemcpyDeviceToDevice));
    cublasInit();
    for (int i=0; i<DATA_SIZE; i++){
	    cutilSafeCall(cudaMemcpy(device_ww2.data, device_save.data, sizeof(float) * device_ww2.row * device_ww2.col, cudaMemcpyDeviceToDevice));
		cublasSgemv('N', device_ww.row, device_ww.col, 2, device_ww.data, device_ww.row,
				device_data.data + i * device_ww.col,
				1,
				-1,
				device_ww2.data,
				1);
		cudaThreadSynchronize();
		cudaError_t lasterror = cudaGetLastError();
		if (lasterror)
			printf("sgemv: %s\n", cudaGetErrorString(lasterror));
    	reduce<<<1,REDUCE_BLOCKSIZE>>>(device_ww_count.data,device_indices.data,device_sum.data, device_ww2.data,device_data.data, i);
    	cudaThreadSynchronize();
    	lasterror = cudaGetLastError();
    	if (lasterror)
        	printf("reduce:%d %s\n", i, cudaGetErrorString(lasterror));
    }

    cublasShutdown();

    //MANUAL: size should corresponde to total number of rows in data...
    buildImage<<<1834,32>>>(device_ret.data,device_labels.data,device_indices.data);
    cudaThreadSynchronize();
    printf("build image %s\n", cudaGetErrorString(cudaGetLastError()));
    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;
    printf("Run time %f\n\n", time);

    cutResetTimer(timer);

    block = dim3(16,16);
    grid = dim3(IMAGE_M/16, IMAGE_N/16);
    cudaMemset(device_scratch.data, 0, sizeof(float) * IMAGE_MxN * VECTOR_SIZE);
	update_weights<<<grid,block>>>(device_sum.data, device_scratch.data, device_ww_count.data, device_ww_count2.data, host_beta[0]);
	cudaThreadSynchronize();
	update_weights2<<<grid,block>>>(device_ww.data, device_sum.data, device_scratch.data, device_ww_count.data, device_ww_count2.data, host_beta[0], host_alpha[0]);

	cudaThreadSynchronize();
//    cutStartTimer(timer);
//    cutilSafeCall(cudaMemcpy(a, device_ww.data, sizeof(float) * IMAGE_MxN * 16, cudaMemcpyDeviceToHost));
//	int ww_count[device_ww.row * device_ww.col];
//	cutilSafeCall(cudaMemcpy(ww_count, device_ww_count.data, sizeof(int) * device_ww.row * device_ww.col, cudaMemcpyDeviceToHost));
//    cutStopTimer(timer);
//    time = cutGetTimerValue(timer);
//    total_time += time;
//    printf("Transfer back time %f\n\n", time);

    block = dim3(16,16);
    grid = dim3(IMAGE_M/16,IMAGE_N/16);
    expandImage<<<grid,block>>>(device_pbo, device_ret.data);

    printf("Total Time: %f\n\n", total_time);
#if DEBUG_PRINT
	for (int i=0; i<16; i++){
		for (int j=0; j<IMAGE_N; j++){
			for (int k=0; k<IMAGE_M; k++){
				printf("%f ", a[i * IMAGE_MxN + j * IMAGE_M + k]);
			}
			printf("\n");
		}
		printf("\n");
	}

	for (int i=0; i<IMAGE_N; i++){
		for (int j=0; j<IMAGE_M;j++){
			printf("%d ", ww_count[i * IMAGE_M + j]);
		}printf("\n");
	}
#endif
	host_r++;
	host_alpha[0] = max(0.01, host_alpha[1] * (1.0 - (host_r/host_T)));
	host_beta[0] = max(0., host_beta[1] - host_r / 1.5);

   	return EXIT_SUCCESS;
}
