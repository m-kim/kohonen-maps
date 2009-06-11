#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>

#define REDUCE_BLOCKSIZE 256
#define LOG2_REDUCE_BLOCKSIZE 8

__global__ void reduce(uint *ret, uint *indices, const float *ww2, int index)
{
	int size = 1024;
	//using shared memory here will limit me to 8KB...
	//initialize with hard coded numbers because compile error on variable initialization
	__shared__ uint mem[1024];
	__shared__ float s_ww2[1024];

	int blocksize = REDUCE_BLOCKSIZE;
	int coalesce_num = size/blocksize;
	mem[threadIdx.x] = threadIdx.x;

	for (int i=0; i<1024/REDUCE_BLOCKSIZE; i++){
		mem[threadIdx.x + i * blocksize] = threadIdx.x + i * blocksize;
		s_ww2[threadIdx.x + i * blocksize] = ww2[threadIdx.x + i * blocksize];
	}


	// 256->32
	for (int j=1; j < coalesce_num; j++){
		if (threadIdx.x + blocksize * j < 896){
			mem[threadIdx.x] = (s_ww2[mem[threadIdx.x]] > s_ww2[mem[j * blocksize + threadIdx.x]])?
														mem[threadIdx.x]:mem[j * blocksize+threadIdx.x];
		}
	}

	//32->16, 16->8, 8->4, 4->2, 2->1
	for (int i=0; i<LOG2_REDUCE_BLOCKSIZE; i++){
		__syncthreads();
		blocksize = blocksize/2;
		__syncthreads();

		if (threadIdx.x < blocksize){
			mem[threadIdx.x] = s_ww2[ mem[blocksize +threadIdx.x]] < s_ww2[mem[threadIdx.x]]? mem[threadIdx.x]:(mem[blocksize+threadIdx.x]);
		}
	}
	__syncthreads();
	ret[ mem[0] ]++;
	indices[index] = mem[0];
}

__global__ void buildImage(uint *im, uint *labels, uint *indices)
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ int nn[32];
	__shared__ int mm[32];
	nn[threadIdx.x] = indices[i] / 28;
	mm[threadIdx.x] = indices[i] - 28 * nn[threadIdx.x];
	im[ nn[threadIdx.x] * 28 + mm[threadIdx.x]] = labels[i] + 1;


}
extern "C" int runCudasGemm(MATRIX ww, MATRIX ww2, MATRIX data)
{
    float* device_A, *device_B, *device_ww2, *device_save;
    float* a = 0;
    a = (float *)malloc (ww.row * data.col * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    unsigned int *device_ret = 0, *device_labels, *device_indices;

    unsigned int *ret = (unsigned int*)malloc(sizeof(unsigned int) * ww.row);
	uint *indices = (uint*)malloc(sizeof(uint) * data.col);

	unsigned int timer;
    cutCreateTimer(&timer);
    double time,total_time;

    total_time = 0;
    cutResetTimer(timer);
    cutStartTimer(timer);

	uint *labels = (uint*)malloc(sizeof(uint) * data.col);
	for (int i=0; i< 20; i++){
		for (int j=0;j<1000; j++){
			labels[i * 1000 + j] = i;
		}
	}

	cutilSafeCall(cudaMalloc((void**)&device_labels, sizeof(uint) * data.col));
	cutilSafeCall(cudaMemcpy(device_labels, labels, sizeof(uint) * data.col, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMalloc((void**)&device_ret, sizeof(unsigned int) * ww.row));
    cutilSafeCall(cudaMemset((void*)device_ret, 0, sizeof(unsigned int) * ww.row));

    cutilSafeCall(cudaMalloc((void**)&device_indices, sizeof(unsigned int) * data.col));
    cutilSafeCall(cudaMemset(device_indices, 0, sizeof(unsigned int) * data.col));

    printf("setup matrix ww %d %d\n", ww.row, ww.col);
    cutilSafeCall(cudaMalloc((void**)&device_A, sizeof(float) * ww.row * ww.col));
    cutilSafeCall(cudaMemcpy(device_A, ww.data, sizeof(float) * ww.row * ww.col, cudaMemcpyHostToDevice));
    for (int i=0; i<ww.row; i++){
    	for (int j=0; j<ww.col; j++){
    		cutilSafeCall(cudaMemcpy(
    				device_A  + (j * ww.row + i),
    				ww.data + (i * ww.col + j), sizeof(float), cudaMemcpyHostToDevice));
    	}
    }

    printf("setup matrix data %d %d\n", data.row, data.col);
    cutilSafeCall(cudaMalloc((void**)&device_B, sizeof(float) * data.row*data.col));
    cutilSafeCall(cudaMemcpy(device_B, data.data, sizeof(float) * data.row * data.col, cudaMemcpyHostToDevice));
    for (int i=0; i<data.row; i++){
    	for (int j=0; j<data.col; j++){
    		cutilSafeCall(cudaMemcpy(
    				device_B  + (j * data.row + i),
    				data.data + (i * data.col + j), sizeof(float), cudaMemcpyHostToDevice));
    	}
    }


    printf("setup vector ww2 %d\n", ww2.row * ww2.col);
    cutilSafeCall(cudaMalloc((void**)&device_ww2, sizeof(float) * ww2.row * ww2.col));
    cutilSafeCall(cudaMemcpy(device_ww2, ww2.data, sizeof(float) * ww2.row * ww2.col, cudaMemcpyHostToDevice));
    printf("setup vector sum %d\n", ww.row);
    cutilSafeCall(cudaMalloc((void**)&device_save, sizeof(float) * ww.row));
    cutilSafeCall(cudaMemcpy(device_save, device_ww2, sizeof(float) * ww.row, cudaMemcpyDeviceToDevice));

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;
    printf("Initialization time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);

    for (int i=0; i<20000; i++){
	    cutilSafeCall(cudaMemcpy(device_ww2, device_save, sizeof(float) * ww.row, cudaMemcpyDeviceToDevice));
		cublasSgemv('N', 896,16, 2, device_A, 896,
				device_B + i * 16,
				1,
				-1,
				device_ww2,
				1);
		cudaThreadSynchronize();
//		cudaError_t lasterror = cudaGetLastError();
//		if (lasterror)
//			printf("sgemv: %s\n", cudaGetErrorString(lasterror));
    	reduce<<<1,REDUCE_BLOCKSIZE>>>(device_ret,device_indices, device_ww2, i);
    	cudaThreadSynchronize();
//    	lasterror = cudaGetLastError();
//    	if (lasterror)
//        	printf("reduce:%d %s\n", i, cudaGetErrorString(lasterror));

    }


    buildImage<<<625,32>>>(device_ret,device_labels,device_indices);
    cudaThreadSynchronize();
    printf("build image %s\n", cudaGetErrorString(cudaGetLastError()));
    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Run time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);
//    cutilSafeCall(cudaMemcpy(a, device_ww2, sizeof(float) * 896, cudaMemcpyDeviceToHost));
//    cutilSafeCall(cudaMemcpy(indices, device_indices, sizeof(uint) * data.col, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(ret, device_ret, sizeof(uint) * 896, cudaMemcpyDeviceToHost));

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Transfer back time %f\n\n", time);

    printf("Total Time: %f\n\n", total_time);
    cudaFree (device_A);
    cudaFree (device_B);
    cudaFree (device_ww2);
	cudaFree(device_ret);
	cudaFree(device_indices);
	cudaFree(device_labels);
//	int counter = 0;
//	for (int i=0; i<56; i++){
//		 for (int j=0; j<16; j++){
//				printf("%d ", ret[i * 16 + j]);
//				counter += ret[i * 16 + j];
//		 }
//		 printf("\n");
//	}
//	printf("%d\n",counter);

//	uint *nn = (uint*)malloc(sizeof(uint) * data.col);
//	uint *mm = (uint*)malloc(sizeof(uint) * data.col);
//	for (int i=0; i<20000; i++){
//		nn[i] = indices[i]/28;
//		mm[i] = indices[i] - 28 * nn[i];
//	}
//
//	for (int i=0; i<data.col; i++)
//	{
//		im[ nn[i] * 28 + mm[i]] = labels[i] + 1;
//	}

	for (int i=0; i<32; i++){
		printf("[");
		for (int j=0; j<28; j++){
			printf("%2d ", ret[i * 28 + j]);
		}
		printf("]\n");
	}
	delete a, ret, indices, labels;
    return EXIT_SUCCESS;
}



