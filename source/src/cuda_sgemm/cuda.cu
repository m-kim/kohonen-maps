#include "shared.h"
#include <cublas.h>
#include <stdio.h>
#include <cutil_inline.h>
#define EPSILON 0.000001

extern "C" int genome_index;

MATRIX<MATRIX_TYPE> device_ww2, device_save, device_sum, device_scratch;
MATRIX<unsigned int> device_labels, device_indices,device_ww_count, device_ret,device_ww_count2;
ORDERED_MATRIX<MATRIX_TYPE, COLUMN_MAJOR> device_ww, device_data;


unsigned int *ret, *indices;

float host_alpha[2];
int host_r = -1, host_beta[2];

__constant__ uint constant_color[COLOR_SIZE];


__global__ void calc_ww2(const MATRIX<MATRIX_TYPE> ww, MATRIX_TYPE *ww2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j=0; j<VECTOR_SIZE; j++){
		//this shouldn't be backwards...*sigh*
		ww2[i] += pow(ww.data[i * ww.row + j ],2);
	}
}

__global__ void update_weights(float *a, float *b, uint *ww_count, uint *count, int _beta)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	int index = row + IMAGE_M * col;
	if (col < IMAGE_M){
		int imin = max(row - _beta, 0);
		int imax = min(row + _beta + 1, IMAGE_N);

		for (int x=imin; x<imax; x++){
			for (int k=0; k<VECTOR_SIZE; k++){
				b[k + VECTOR_SIZE * ( row + IMAGE_M * col )] += a[k + VECTOR_SIZE * (x + IMAGE_M * col)];
			}
			count[index] += ww_count[x + IMAGE_M * col];
		}

//		for (int i=0; i<VECTOR_SIZE; i++){  //vector size...
//			for (int k= _min; k<_max; k++){
//				b[i * IMAGE_MxN + index]  += a[i * IMAGE_MxN + k * IMAGE_M + col];
//			}
//		}
//		for (int k= _min; k<_max; k++){
//			count[index] += ww_count[k * IMAGE_M + col];
//		}
	}
}

__global__ void update_weights2(float *ww, float *a, float *b, uint *ww_count, uint *count, int _beta, float _alpha)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	int index = row * IMAGE_M + col;
//	__shared__ float s_ww[IMAGE_N * IMAGE_M];
	int _min = max(col - _beta, 0);
	int _max = min(col + _beta + 1, IMAGE_M);

	if (col < IMAGE_M){
		int imin = max(row - _beta,0);
		int imax = min(row + _beta + 1,IMAGE_N);
		float sum = 0;
		for (int x=imin; x<imax; x++){
			for (int k=0; k<VECTOR_SIZE; k++){
				a[k + VECTOR_SIZE * ( col + IMAGE_M * row) ] += b[k + VECTOR_SIZE * ( col + IMAGE_M * x) ];
			}
			ww_count[col + IMAGE_M * row] += count[col + IMAGE_M * x];
		}
	}
//		for (int i=0; i<VECTOR_SIZE; i++){  //vector size...
//			for (int k= _min; k<_max; k++){
//				a[i * IMAGE_MxN + index]  += b[i * IMAGE_MxN + row * IMAGE_M + k];
//			}
//		}
//		for (int k= _min; k<_max; k++){
//			ww_count[index] += count[row * IMAGE_M + k];
//		}
//
//		for (int i=0; i<VECTOR_SIZE; i++){
//			if (ww_count[index] == 0)
//				a[i * IMAGE_MxN + index] = 0;
//			else
//				a[ i * IMAGE_MxN + index]  /= (float)ww_count[index];
//        	ww[i * IMAGE_MxN + index] = abs(ww[i * IMAGE_MxN + index]  +_alpha * (a[i * IMAGE_MxN + index] - ww[i * IMAGE_MxN + index]));
//		}
//		for (int i=0; i<VECTOR_SIZE; i++){
//	    	ww[index] += ww[i * IMAGE_MxN + index];
//		}
//		for (int i=0; i<VECTOR_SIZE; i++){
//			if (ww[index] > 0)
//				ww[i * IMAGE_MxN + index] /= (float)ww[index];
//			else
//				ww[i * IMAGE_MxN + index] = 0;
//
//		}
//	}
//	__syncthreads();
//	if (col < IMAGE_M){
//		for (int i=0; i<VECTOR_SIZE; i++){
//			if (ww[index] > 0)
//				ww[i * IMAGE_MxN + index] /= (float)ww[index];
//			else
//				ww[i * IMAGE_MxN + index] = 0;
//
//		}
//	}
}

//Calculate argmax and sum the data vectors
__global__ void reduce(uint *ret, uint *indices, float *ww_sum, const float *vec, const float *data, int index)
{
	int size = 1024;
	//using shared memory here will limit me...
	//initialize with hard coded numbers because compile error on variable initialization
	__shared__ int argmax[1024];
	__shared__ float s_vec[1024];

	int blocksize = REDUCE_BLOCKSIZE;
	int coalesce_num = size/blocksize;

	for (int i=0; i<1024/REDUCE_BLOCKSIZE; i++){
		argmax[threadIdx.x + i * blocksize] = threadIdx.x + i * blocksize;
		s_vec[threadIdx.x + i * blocksize] = vec[threadIdx.x + i * blocksize];
	}


	// Large number ->32
	for (int j=1; j < coalesce_num; j++){
		if (threadIdx.x + blocksize * j < IMAGE_MxN){
			argmax[threadIdx.x] = (s_vec[argmax[threadIdx.x]] > s_vec[argmax[j * blocksize + threadIdx.x]])?
						argmax[threadIdx.x]:argmax[j * blocksize + threadIdx.x];
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
	if (threadIdx.x < 1){
		ret[ argmax[0] ]++;
		indices[index] = argmax[0];
	}
	//take the vector from data and save it to ww_sum
	if (threadIdx.x < VECTOR_SIZE)
		ww_sum[ argmax[0] *VECTOR_SIZE + threadIdx.x] += data[index * VECTOR_SIZE + threadIdx.x];//ww_sum[ 410 * 16 + threadIdx.x] = data[threadIdx.x];
}

__global__ void buildImage(uint *im, uint *labels, uint *indices)
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ int nn[IMAGE_N];
	__shared__ int mm[IMAGE_N];
	nn[threadIdx.x] = indices[i] / IMAGE_M;
	mm[threadIdx.x] = indices[i] - IMAGE_M * nn[threadIdx.x];
	im[ nn[threadIdx.x] * IMAGE_M + mm[threadIdx.x]] = labels[i];
}

__global__ void buildSplitImage(uint *im, uint *labels, uint *indices, int g_index)
{
	uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
	uint tidy = threadIdx.y + blockDim.y * blockIdx.y;
	uint index = tidx * IMAGE_N + tidy;

	int genome[GENOMIC_DATA_COUNT];

	for (int i=0; i<GENOMIC_DATA_COUNT; i++)
		genome[i] = 0;

	for (int i=0; i<DATA_SIZE; i++){
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
	im[index] = 0;
//	if (genome[0] > genome[1] && genome[0] > genome[2] && genome[0] > genome[3])
//		im[index] = genome[g_index];
//	else if (genome[1] > genome[0] && genome[1] > genome[2] && genome[1] > genome[3])
//		im[index] = genome[g_index];
//	else if (genome[2] > genome[0] && genome[2] > genome[1] && genome[2] > genome[3])
//		im[index] = genome[g_index];
//	else if (genome[3] > genome[0] && genome[3] > genome[1] && genome[3] > genome[2])
//		im[index] = genome[g_index];
//	else
//		im[index] = 0;
}

__global__ void expandSplitImage(uint *im, const uint *ret)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<16; i++){
		for (int j=0; j<16; j++){
			im[(y * 16 + j) * 512 + x * 16 + i] = LUMINANCE_ADJUSTMENT * ret[y * IMAGE_M + x];
		}
	}
}

__global__ void expandLogImage(unsigned char *im, const uint *ret)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<16; i++){
		for (int j=0; j<16; j++){
			im[(y * 16 + j) * 512 + x * 16 + i] = 10 * logf(ret[y * IMAGE_M + x]);
		}
	}
}
__global__ void expandConstantImage(uint *im, const uint *ret)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	for (int i=0; i<16; i++){
		for (int j=0; j<16; j++){
			im[(y * 16 + j) * 512 + x * 16 + i] = constant_color[ret[y * IMAGE_M + x]] * ret[y * IMAGE_M + x];
		}
	}
}

extern "C" void generateSplitImage(int g_index, unsigned int * device_split_pbo)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_M/16,IMAGE_N/16);
	expandSplitImage<<<grid,block>>>(device_split_pbo, device_ret.data + g_index * IMAGE_MxN);
}

extern "C" void cleanup()
{
    cudaFree (device_ww.data);
    cudaFree (device_data.data);
    cudaFree (device_ww2.data);
	cudaFree(device_indices.data);
	cudaFree(device_labels.data);
	delete  ret, indices;
}
extern "C" void setupCuda(ORDERED_MATRIX<MATRIX_TYPE, COLUMN_MAJOR> ww,  ORDERED_MATRIX<MATRIX_TYPE, ROW_MAJOR> data, uint *labels, unsigned int *device_regular_pbo, uint *device_split_pbo, unsigned char *device_log_pbo)
{
    //setup color
	unsigned char color[COLOR_SIZE * 4];
	for(unsigned int i=0; i<COLOR_SIZE * 4; i+=4){
		color[i + 1] = (unsigned char)i;
		color[i + 2] = (i + 64) % 256;
		color[i + 3] = (i + 128) % 256;
		color[i] = (i + 192) % 256;
	}

	color[0] = 0;
	color[1] = 0;
	color[2] = 0;
	color[3] = 0;

	color[4] = 255;
	color[5] = 0;
	color[6] = 0;
	color[7] = 0;

	color[8] = 0;
	color[9] = 255;
	color[10] = 0;
	color[11] = 0;

	color[12] = 0;
	color[13] = 0;
	color[14] = 255;
	color[15] = 0;

	color[16] = 255;
	color[17] = 255;
	color[18] = 0;
	color[19] = 0;

	color[20] = 255;
	color[21] = 0;
	color[22] = 255;
	color[23] = 0;

	color[24] = 0;
	color[25] = 255;
	color[26] = 255;
	color[27] = 0;

	color[28] = 128;
	color[29] = 128;
	color[30] = 128;
	color[31] = 0;

	color[32] = 255;
	color[33] = 255;
	color[34] = 255;
	color[35] = 0;

	cutilSafeCall(cudaMemcpyToSymbol(constant_color, color, sizeof(unsigned int) * COLOR_SIZE, 0));

	host_beta[0] = 8;
	host_beta[1] = 8;
	host_alpha[0] = .6;
	host_alpha[1] = .6;


	cudaMemset(device_regular_pbo, 128, sizeof(unsigned int) * 512 * 512);
	cudaMemset(device_split_pbo, 128, sizeof(unsigned int) * 512 * 512);
	cudaMemset(device_log_pbo, 128, sizeof(unsigned char) * 512 * 512);

	device_labels.row = data.row;
	device_labels.col = 1;
	cutilSafeCall(cudaMalloc((void**)&device_labels.data, sizeof(uint) * data.row));
	cutilSafeCall(cudaMemcpy(device_labels.data, labels, sizeof(uint) * data.row, cudaMemcpyHostToDevice));

	device_ww_count.row = 1024;
	device_ww_count.col = 1;
	cutilSafeCall(cudaMalloc((void**)&device_ww_count.data, sizeof(unsigned int) * device_ww_count.row));
    cutilSafeCall(cudaMemset((void*)device_ww_count.data, 0, sizeof(unsigned int) * device_ww_count.row));

	device_ww_count2.row = 1024;
	device_ww_count2.col = 1;
	cutilSafeCall(cudaMalloc((void**)&device_ww_count2.data, sizeof(unsigned int) * device_ww_count2.row));
    cutilSafeCall(cudaMemset((void*)device_ww_count2.data, 0, sizeof(unsigned int) * device_ww_count2.row));

    //multiply by the number of genomes
    //+1 for the regular image
    device_ret.row = ww.row * (GENOMIC_DATA_COUNT + 1);
    device_ret.col = 1;
	cutilSafeCall(cudaMalloc((void**)&device_ret.data, sizeof(unsigned int) * device_ret.row));
    cutilSafeCall(cudaMemset((void*)device_ret.data, 0, sizeof(unsigned int) * device_ret.row));

    device_indices.row = DATA_SIZE;
    device_indices.col = 1;
    cutilSafeCall(cudaMalloc((void**)&device_indices.data, sizeof(unsigned int) * DATA_SIZE));
    cutilSafeCall(cudaMemset((void*)device_indices.data, 0, sizeof(unsigned int) * DATA_SIZE));

    device_ww.row = ww.row;
    device_ww.col = ww.col;
    printf("setup matrix ww %d %d\n", ww.row, ww.col);
    cutilSafeCall(cudaMalloc((void**)&device_ww.data, sizeof(float) * ww.row * ww.col));
    cutilSafeCall(cudaMemcpy(device_ww.data, ww.data, sizeof(float) * ww.row * ww.col, cudaMemcpyHostToDevice));

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


	ret = (unsigned int*)malloc(sizeof(unsigned int) * ww.row);
	indices = (uint*)malloc(sizeof(uint) * data.row);

    device_data.row = data.row;
    device_data.col = data.col;
    printf("setup matrix data %d %d\n", device_data.row, device_data.col);
    cutilSafeCall(cudaMalloc((void**)&device_data.data, sizeof(float) * device_data.row*device_data.col));
    cutilSafeCall(cudaMemcpy(device_data.data, data.data, sizeof(float) * device_data.row * device_data.col, cudaMemcpyHostToDevice));
}

extern "C" int runCuda(unsigned int *device_regular_pbo, unsigned int *device_split_pbo, unsigned char *device_log_pbo)
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
    cutilSafeCall(cudaMemset((void*)device_ww_count2.data, 0, sizeof(unsigned int) * device_ww_count2.row));
    cutilSafeCall(cudaMemset(device_ww2.data, 0, sizeof(float) * device_ww2.row * device_ww2.col));
    cutilSafeCall(cudaMemset(device_ret.data, 0, sizeof(unsigned int) * device_ret.row));


    //this is related to IMAGE_MXN
    calc_ww2<<<IMAGE_MxN/128,128>>>(device_ww,device_ww2.data);
    cudaThreadSynchronize();

    cutilSafeCall(cudaMemcpy(device_save.data, device_ww2.data, sizeof(float) * device_ww2.row * device_ww2.col, cudaMemcpyDeviceToDevice));
    cublasInit();
    for (int i=0; i<DATA_SIZE; i++){
    	if ( !(i % 10000) )
    		printf("%d\n",i);
	    cutilSafeCall(cudaMemcpy(device_ww2.data, device_save.data, sizeof(float) * device_ww2.row * device_ww2.col, cudaMemcpyDeviceToDevice));
		cublasSgemv('T', device_ww.row, device_ww.col, 2, device_ww.data, device_ww.row,
				device_data.data + i * device_data.col,
				1,
				-1,
				device_ww2.data,
				1);
		cudaThreadSynchronize();

		cudaError_t lasterror = cudaGetLastError();
		if (lasterror)
			printf("sgemv: %s\n", cudaGetErrorString(lasterror));

		//the device_ww_count that's returned *might* be transposed.  Right now, the data is correct, but might need tranposing.
    	reduce<<<1,REDUCE_BLOCKSIZE>>>(device_ww_count.data,device_indices.data,device_sum.data, device_ww2.data,device_data.data, i);
    	cudaThreadSynchronize();
    	lasterror = cudaGetLastError();
    	if (lasterror)
        	printf("reduce:%d %s\n", i, cudaGetErrorString(lasterror));
    }

    cublasShutdown();

    ORDERED_MATRIX<unsigned int, COLUMN_MAJOR> count;
    count.row = 32;
    count.col = 32;
    count.data = (unsigned int*)malloc(count.row * count.col * sizeof(unsigned int));
	cudaMemcpy(count.data, device_ww_count.data, device_ww_count.row * device_ww_count.col * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	ORDERED_MATRIX<int, COLUMN_MAJOR> cnt;
	cnt.row = 32;
	cnt.col = 32;
	cnt.data = (int*)malloc(cnt.row * cnt.col * sizeof(int));
	memset(cnt.data, 0, sizeof(int) * cnt.row * cnt.col);

	ORDERED_MATRIX<float, COLUMN_MAJOR> argh;
	argh.row = device_sum.row;
	argh.col = device_sum.col;
	argh.data = (float*)malloc(argh.row * argh.col * sizeof(float));
	cudaMemcpy(argh.data, device_sum.data, sizeof(float) * argh.col * argh.row, cudaMemcpyDeviceToHost);

	ORDERED_MATRIX<float, COLUMN_MAJOR> cc_sum;
	cc_sum.row = 4;
	cc_sum.col = 1024;
	cc_sum.data = (float*)malloc(cc_sum.row * cc_sum.col *sizeof(float));
	memset(cc_sum.data, 0 , sizeof(float) * cc_sum.row * cc_sum.col);
	for (int i=0; i<32; i++){
		int imin = max(i - host_beta[0],0);
		int imax = min(i+ host_beta[0] + 1,IMAGE_N);

		for (int j=0; j<32; j++){
			for (int x=imin; x<imax; x++){
				for (int k=0; k<4; k++){
					cc_sum(k, i + IMAGE_M * j) += argh(k, x + IMAGE_M * j);
				}
				cnt(i,j) += count(x,j);
			}
		}
	}

	memset(argh.data, 0, sizeof(float) * argh.row * argh.col);
	memset(count.data, 0, sizeof(int) * count.row * count.col);

	for (int i=0; i<32; i++){
		int imin = max(i - host_beta[0],0);
		int imax = min(i+ host_beta[0] + 1,IMAGE_N);
		for (int j=0; j<32; j++){
			float sum = 0;
			for (int x=imin; x<imax; x++){
				for (int k=0; k<4; k++){
					argh(k, i + IMAGE_M * j) += cc_sum(k, j + IMAGE_M * x);
				}
				count(j,i) += cnt(j,x);
			}
		}
	}

	cudaMemset(device_scratch.data, 0, sizeof(float) * IMAGE_MxN * VECTOR_SIZE);
	grid = dim3(2,2);
	block = dim3(16,16);
	update_weights<<<grid,block>>>(device_sum.data, device_scratch.data, device_ww_count.data, device_ww_count2.data, host_beta[0]);
	cudaThreadSynchronize();

	update_weights2<<<grid,block>>>(device_ww.data, device_sum.data, device_scratch.data, device_ww_count.data, device_ww_count2.data, host_beta[0], host_alpha[0]);
	cudaThreadSynchronize();

	cutilSafeCall(cudaMemcpy(cc_sum.data, device_sum.data, cc_sum.row * cc_sum.col * sizeof(float), cudaMemcpyDeviceToHost));
		for (int i=0; i<4; i++){
			for (int j=0; j<32; j++){
				for (int k=0; k<32; k++){
					printf("%f ", cc_sum(i, j + IMAGE_M * k));
				}
				printf("\n");
			}
		}

    cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;
    printf("Run time %f\n\n", time);
    cutResetTimer(timer);

    block = dim3(16,16);
    grid = dim3(IMAGE_M/16, IMAGE_N/16);
    buildImage<<<BUILD_IMAGE_GRID_SIZE,32>>>(device_ret.data + GENOMIC_DATA_COUNT * IMAGE_MxN,
    											device_labels.data,device_indices.data);
    for (int i=0; i<GENOMIC_DATA_COUNT; i++)
    	buildSplitImage<<<grid,block>>>(device_ret.data + i * IMAGE_MxN,device_labels.data,device_indices.data,i);

    expandConstantImage<<<grid,block>>>(device_regular_pbo, device_ret.data + GENOMIC_DATA_COUNT * IMAGE_MxN);
	expandLogImage<<<grid,block>>>(device_log_pbo, device_ww_count.data + GENOMIC_DATA_COUNT * IMAGE_MxN);
	generateSplitImage(genome_index, device_split_pbo);


    printf("Total Time: %f\n\n", total_time);
#if DEBUG_PRINT
    unsigned char count[262144];
    cutilSafeCall(cudaMemcpy(count, device_log_pbo, sizeof(unsigned char) * 262144, cudaMemcpyDeviceToHost));
	int counter = 0;
	for (int i=0; i<512; i++){
		for (int j=0; j<512; j++){
			printf("%d ", count[i * 512 + j]);
		}
		printf("\n");
	}

#endif
	host_r++;
	host_alpha[0] = max(0.01, host_alpha[1] * (1.0 - ((float)host_r/host_T)));
	host_beta[0] = max(0., host_beta[1] - host_r / 1.5);

	printf("r: %d alpha %f: beta %d\n", host_r, host_alpha[0], host_beta[0]);
   	return EXIT_SUCCESS;
}
