#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>

//there's probably a much nicer way to do this...
//but lets try this for now
__global__ void coalesce(const float *ww, const float *data, const float *ww2,
						unsigned int *ret,
						unsigned int *indices,
						float alpha, float beta,
						int M, int N, int K)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	float sum = 0;
	float max_val = -10000;
	int argmax = 0;
	if (i < 20000){
		for (int j=0; j<M; j++){
			sum = 0;
			for (int k=0; k<K; k++){
				sum += ww[j * K + k] * data[k * N + i];
			}
			sum = alpha * sum + beta * ww2[j];
			//ww2[j * N + i] = sum;
			if (max_val < sum){
					argmax = j;
					max_val = sum;
			}
		}
		//Quadro FX 5600 is not compute 1.1 compatible...*sigh*
		//this might not matter because atomics are only block level
		ret[argmax]++;
		indices[i] = argmax;
	}
}

__global__ void reduce(uint *ret, float *ww2)
{
//	float max_val = -10000;
//	int argmax = 0;
//	for (int i=0; i<896; i++){
//		if (max_val < ww2[i]){
//			max_val = ww2[i];
//			argmax = i;
//		}
//	}
//	ret[argmax]++;

	int size = 1024;
	//using shared memory here will limit me to 8KB...
	//initialize with hard coded numbers because compile error on variable initialization
	__shared__ uint mem[1024];
	__shared__ uint s_ww2[1024];

	int blocksize = 32;
	int coalesce_num = size/blocksize;
	mem[threadIdx.x] = threadIdx.x;

	//32 threads running...so multiply by 32 = 1024
	//this is not a coalesced memory access...
	for (int i=0; i<32; i++){
		mem[threadIdx.x + i * blocksize] = threadIdx.x + i * blocksize;
		s_ww2[threadIdx.x + i * blocksize] = ww2[threadIdx.x + blocksize * i];
	}


	// 256->32
	for (int j=1; j < coalesce_num; j++){
		if (threadIdx.x + blocksize * j < 896){
			mem[threadIdx.x] = (ww2[mem[threadIdx.x]] > ww2[mem[j * blocksize + threadIdx.x]])?
														mem[threadIdx.x]:mem[j * blocksize+threadIdx.x];
		}
	}

////	//32->16, 16->8, 8->4, 4->2, 2->1
	for (int i=0; i<5; i++){
		__syncthreads();
		blocksize = blocksize/2;
		__syncthreads();

		if (threadIdx.x < blocksize){
			mem[threadIdx.x] = ww2[  mem[blocksize +threadIdx.x]] < ww2[mem[threadIdx.x]]? mem[threadIdx.x]:(mem[blocksize+threadIdx.x]);
		}
	}
	ret[ mem[0]]++;
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

    unsigned int *device_ret = 0;
    unsigned int *ret = (unsigned int*)malloc(sizeof(unsigned int) * ww.row);

    unsigned int *device_indices = 0;

    unsigned int timer;
    cutCreateTimer(&timer);
    double time,total_time;

    total_time = 0;
    cutResetTimer(timer);
    cutStartTimer(timer);

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

//    coalesce<<<80,256>>>(device_A, device_B, device_ww2, device_ret, device_indices, 2.0, -1.0, ww.row,data.col, data.row);
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
    	reduce<<<1,32>>>(device_ret, device_ww2);
    	cudaThreadSynchronize();
//    	lasterror = cudaGetLastError();
//    	if (lasterror)
//        	printf("reduce:%d %s\n", i, cudaGetErrorString(lasterror));

    }


    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    printf("Run time %f\n\n", time);

    cutResetTimer(timer);
    cutStartTimer(timer);
    cutilSafeCall(cudaMemcpy(a, device_ww2, sizeof(float) * 896, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(ret, device_ret, sizeof(float) * 896, cudaMemcpyDeviceToHost));
    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;

    for (int i=0; i<16; i++)
    	printf("%f ", a[i+864]);
    printf("\n");
    printf("Transfer back time %f\n\n", time);

    printf("Total Time: %f\n\n", total_time);
    cudaFree (device_A);
    cudaFree (device_B);
    cudaFree (device_ww2);
	cudaFree(device_ret);
	cudaFree(device_indices);


	for (int i=0; i<896; i++){
		if (i < 5 || i > 891)
			printf("%f ", a[i]);
	}
	printf("\n");

	int counter = 0;
	for (int i=0; i<56; i++){
		 for (int j=0; j<16; j++){
				printf("%d ", ret[i * 16 + j]);
				counter += ret[i * 16 + j];
		 }
		 printf("\n");
	}
	printf("%d\n",counter);


	delete a, ret;
    return EXIT_SUCCESS;
}



