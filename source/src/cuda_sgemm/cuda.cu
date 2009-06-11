#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>

#define REDUCE_BLOCKSIZE 256
#define LOG2_REDUCE_BLOCKSIZE 8

MATRIXf device_A, device_B, device_ww2, device_save;
MATRIXu device_labels, device_indices, device_ww_count, device_ret;

float* a;
unsigned int *ret, *indices;


__constant__ uint constant_color[256];

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

__global__ void expandImage(uint *im, const uint *ret)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<16; i++){
		for (int j=0; j<16; j++){
			im[(y * 16 + j) * 512 + x * 16 + i] = constant_color[ret[y * 28 + x]] * ret[y * 28 + x];
		}
	}
}

extern "C" void cleanup()
{
    cudaFree (device_A.data);
    cudaFree (device_B.data);
    cudaFree (device_ww2.data);
	cudaFree(device_indices.data);
	cudaFree(device_labels.data);
	delete a, ret, indices;
}
extern "C" void setupCuda(MATRIXf ww, MATRIXf ww2, MATRIXf data, uint *labels, unsigned int *device_pbo)
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

	cudaMemset(device_pbo, 128, sizeof(unsigned int) * 512 * 512);

	device_labels.row = data.col;
	device_labels.col = 1;
	cutilSafeCall(cudaMalloc((void**)&device_labels.data, sizeof(uint) * data.col));
	cutilSafeCall(cudaMemcpy(device_labels.data, labels, sizeof(uint) * data.col, cudaMemcpyHostToDevice));

	device_ww_count.row = ww.row;
	device_ww_count.col = ww.col;
	cutilSafeCall(cudaMalloc((void**)&device_ww_count.data, sizeof(unsigned int) * ww.row));
    cutilSafeCall(cudaMemset((void*)device_ww_count.data, 0, sizeof(unsigned int) * ww.row));

    device_indices.row = ww.row;
    device_indices.col = 1;
	cutilSafeCall(cudaMalloc((void**)&device_ret.data, sizeof(unsigned int) * ww.row));
    cutilSafeCall(cudaMemset((void*)device_ret.data, 0, sizeof(unsigned int) * ww.row));

    device_indices.row = data.col;
    device_indices.col = 1;
    cutilSafeCall(cudaMalloc((void**)&device_indices.data, sizeof(unsigned int) * data.col));
    cutilSafeCall(cudaMemset(device_indices.data, 0, sizeof(unsigned int) * data.col));

    device_A.row = ww.row;
    device_A.col = ww.col;
    printf("setup matrix ww %d %d\n", ww.row, ww.col);
    cutilSafeCall(cudaMalloc((void**)&device_A.data, sizeof(float) * ww.row * ww.col));
    cutilSafeCall(cudaMemcpy(device_A.data, ww.data, sizeof(float) * ww.row * ww.col, cudaMemcpyHostToDevice));
    for (int i=0; i<ww.row; i++){
    	for (int j=0; j<ww.col; j++){
    		cutilSafeCall(cudaMemcpy(
    				device_A.data  + (j * ww.row + i),
    				ww.data + (i * ww.col + j), sizeof(float), cudaMemcpyHostToDevice));
    	}
    }

    device_B.row = data.row;
    device_B.col = data.col;
    printf("setup matrix data %d %d\n", data.row, data.col);
    cutilSafeCall(cudaMalloc((void**)&device_B.data, sizeof(float) * data.row*data.col));
    cutilSafeCall(cudaMemcpy(device_B.data, data.data, sizeof(float) * data.row * data.col, cudaMemcpyHostToDevice));
    for (int i=0; i<data.row; i++){
    	for (int j=0; j<data.col; j++){
    		cutilSafeCall(cudaMemcpy(
    				device_B.data  + (j * data.row + i),
    				data.data + (i * data.col + j), sizeof(float), cudaMemcpyHostToDevice));
    	}
    }

	device_ww2.row = ww2.row;
	device_ww2.col = ww2.col;
    printf("setup vector ww2 %d\n", device_ww2.row * device_ww2.col);
    cutilSafeCall(cudaMalloc((void**)&device_ww2.data, sizeof(float) * device_ww2.row * device_ww2.col));
    cutilSafeCall(cudaMemcpy(device_ww2.data, ww2.data, sizeof(float) * ww2.row * ww2.col, cudaMemcpyHostToDevice));

    device_save.row = ww.row;
    device_save.col = ww.col;
    printf("setup vector sum %d\n", ww.row);
    cutilSafeCall(cudaMalloc((void**)&device_save.data, sizeof(float) * ww.row));
    cutilSafeCall(cudaMemcpy(device_save.data, device_ww2.data, sizeof(float) * ww.row, cudaMemcpyDeviceToDevice));


    a = (float *)malloc (ww.row * data.col * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
    }

	ret = (unsigned int*)malloc(sizeof(unsigned int) * ww.row);
	indices = (uint*)malloc(sizeof(uint) * data.col);

}

extern "C" int runCudasGemm(unsigned int *device_pbo)
{


	unsigned int timer;
    cutCreateTimer(&timer);
    double time,total_time;

    total_time = 0;
    cutResetTimer(timer);
    cutStartTimer(timer);



    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;
    printf("Initialization time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);

    for (int i=0; i<20000; i++){
	    cutilSafeCall(cudaMemcpy(device_ww2.data, device_save.data, sizeof(float) * device_ww2.row * device_ww2.col, cudaMemcpyDeviceToDevice));
		cublasSgemv('N', 896,16, 2, device_A.data, 896,
				device_B.data + i * 16,
				1,
				-1,
				device_ww2.data,
				1);
		cudaThreadSynchronize();
//		cudaError_t lasterror = cudaGetLastError();
//		if (lasterror)
//			printf("sgemv: %s\n", cudaGetErrorString(lasterror));
    	reduce<<<1,REDUCE_BLOCKSIZE>>>(device_ww_count.data,device_indices.data, device_ww2.data, i);
    	cudaThreadSynchronize();
//    	lasterror = cudaGetLastError();
//    	if (lasterror)
//        	printf("reduce:%d %s\n", i, cudaGetErrorString(lasterror));
    }

    //once we reach this point, we only really care about the device_indices
    //so we can map 'em to a picture

    buildImage<<<625,32>>>(device_ret.data,device_labels.data,device_indices.data);
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
	cutilSafeCall(cudaMemcpy(ret, device_ret.data, sizeof(uint) * 896, cudaMemcpyDeviceToHost));

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    dim3 block(16,16);
    dim3 grid(2,2);
    expandImage<<<grid,block>>>(device_pbo, device_ret.data);
    printf("Transfer back time %f\n\n", time);

    printf("Total Time: %f\n\n", total_time);
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

	for (int i=0; i<28; i++){
		printf("[");
		for (int j=0; j<28; j++){
			printf("%2d ", ret[i * 28 + j]);
		}
		printf("]\n");
	}

    return EXIT_SUCCESS;
}



