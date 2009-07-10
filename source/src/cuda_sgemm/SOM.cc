#include "SOM.h"
#include  <cublas.h>
#include <iostream>
#include <fstream>
#include <sstream>

extern "C" void expandSplitImage(uint *im, const uint *ret);
extern "C" void prepSum(float *a, float *b, uint *ww_count, uint *count, int _beta);
extern "C" void prepSum2(float *ww, float *a, float *b, uint *ww_count, uint *count, int _beta);
extern "C" void normalizeSum(float *a, unsigned int* ww_count);
extern "C" void calc_ww2(MATRIX_TYPE *ww, MATRIX_TYPE *ww2);
extern "C" void safeMemset(void *ptr, char value, unsigned int size);
extern "C" void buildImage(uint *im, uint *labels, uint *indices);
extern "C" void expandConstantImage(uint *im, const uint *ret);
extern "C" void reduce(uint *ret, uint *indices, float *ww_sum, const float *vec, const float *data, int index);
extern "C" void cuda_updateWeights(float *ww, float *avg_weight, float alpha);
extern "C" void setup(int VECTOR_SIZE, int DATA_SIZE);

SOM::SOM()
{
	host_r = -1;
	VECTOR_SIZE = 2;
	BETA = 4;
	ALPHA = .6;
	host_T = 10;
	genome_index = 0;
	DEBUG_PRINT = 0;
	RUN_CYCLE = 1;
	RUN_DISPLAY = 1;

	DATA_PATH = "/data/";
	DATA_FILE = "data";

	counter = 0;

}
void SOM::generateSplitImage(int g_index, unsigned int * device_split_pbo)
{
	dim3 block(16,16);
	dim3 grid(IMAGE_M/16,IMAGE_N/16);
	expandSplitImage(device_split_pbo, device_ret.data + g_index * IMAGE_MxN);
}

void SOM::updateConvergence()
{
	host_r++;
	host_alpha[0] = std::max(0.01, host_alpha[1] * (1.0 - ((float)host_r/host_T)));
	host_beta[0] = std::max(0., host_beta[1] - host_r / 1.5);
}

void SOM::updateWeights()
{
    //update_weights
    cutilSafeCall(cudaMemset(device_scratch.data, 0, sizeof(float) * IMAGE_MxN * VECTOR_SIZE));

    prepSum(device_sum.data, device_scratch.data, device_ww_count.data, device_ww_count2.data, host_beta[0]);
	prepSum2(device_ww.data, device_sum.data, device_scratch.data, device_ww_count.data, device_ww_count2.data, host_beta[0]);
	normalizeSum(device_sum.data, device_ww_count.data);
	cuda_updateWeights(device_ww.data, device_sum.data, host_alpha[0]);
}

void SOM::setupCuda(ORDERED_MATRIX<MATRIX_TYPE, COLUMN_MAJOR> &ww,
		ORDERED_MATRIX<MATRIX_TYPE, ROW_MAJOR> &data,
		uint *labels,
		unsigned int *device_regular_pbo,
		uint *device_split_pbo,
		unsigned char *device_log_pbo)
{

	host_beta[0] = BETA;
	host_beta[1] = BETA;
	host_alpha[0] = ALPHA;
	host_alpha[1] = ALPHA;

	setup(VECTOR_SIZE,DATA_SIZE);


	cutilSafeCall(cudaMemset(device_regular_pbo, 0, sizeof(unsigned int) * 512 * 512));
	cutilSafeCall(cudaMemset(device_split_pbo, 0, sizeof(unsigned int) * 512 * 512));
	cutilSafeCall(cudaMemset(device_log_pbo, 0, sizeof(unsigned char) * 512 * 512));

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
    if (DEBUG_PRINT)
    	printf("setup matrix ww %d %d\n", ww.row, ww.col);
    cutilSafeCall(cudaMalloc((void**)&device_ww.data, sizeof(float) * ww.row * ww.col));
    cutilSafeCall(cudaMemcpy(device_ww.data, ww.data, sizeof(float) * ww.row * ww.col, cudaMemcpyHostToDevice));

	device_ww2.row = IMAGE_N;
	device_ww2.col = IMAGE_M;
	if (DEBUG_PRINT)
    	printf("setup matrix ww2 %d %d\n", device_ww2.row, device_ww2.col);
    cutilSafeCall(cudaMalloc((void**)&device_ww2.data, sizeof(float) * device_ww2.row * device_ww2.col));
	cutilSafeCall(cudaMemset(device_ww2.data, 0, sizeof(float) * device_ww2.row * device_ww2.col));

    device_sum.row = ww.row;
    device_sum.col = ww.col;
    if (DEBUG_PRINT)
    	printf("setup matrix sum %d %d\n", device_sum.row, device_sum.col);
    cutilSafeCall(cudaMalloc((void**)&device_sum.data, sizeof(float) * device_sum.row * device_sum.col));
    cutilSafeCall(cudaMemset(device_sum.data, 0, sizeof(float) * device_sum.row * device_sum.col ));

	if(DEBUG_PRINT)
    	printf("setup matrix scratch %d %d\n", IMAGE_MxN, VECTOR_SIZE);
    cutilSafeCall(cudaMalloc((void**)&device_scratch.data, sizeof(float) * IMAGE_MxN * VECTOR_SIZE));
    cutilSafeCall(cudaMemset(device_scratch.data, 0, sizeof(float) * IMAGE_MxN * VECTOR_SIZE));

    device_save.row = device_ww2.row;
    device_save.col = device_ww2.col;
    device_save.data = device_scratch.data;

    device_data.row = data.row;
    device_data.col = data.col;
    if (DEBUG_PRINT)
    	printf("setup matrix data %d %d\n", device_data.row, device_data.col);
    cutilSafeCall(cudaMalloc((void**)&device_data.data, sizeof(float) * device_data.row*device_data.col));
    cutilSafeCall(cudaMemcpy(device_data.data, data.data, sizeof(float) * device_data.row * device_data.col, cudaMemcpyHostToDevice));

    updateConvergence();
}

int SOM::runCuda(unsigned int *device_regular_pbo, unsigned int *device_split_pbo, unsigned char *device_log_pbo)
{
	if (DEBUG_PRINT)
		printf("r: %d alpha %f: beta %d\n", host_r, host_alpha[0], host_beta[0]);

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
    calc_ww2(device_ww.data,device_ww2.data);

    cutilSafeCall(cudaMemcpy(device_save.data, device_ww2.data, sizeof(float) * device_ww2.row * device_ww2.col, cudaMemcpyDeviceToDevice));
    cublasInit();
    for (int i=0; i<DATA_SIZE; i++){
    	if ( !(i % 10000) && DEBUG_PRINT)
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
    	reduce(device_ww_count.data,device_indices.data,device_sum.data, device_ww2.data,device_data.data, i);
    	cudaThreadSynchronize();
    	lasterror = cudaGetLastError();
    	if (lasterror)
        	printf("reduce:%d %s\n", i, cudaGetErrorString(lasterror));
    }


    cublasShutdown();

	cutStopTimer(timer);
    time = cutGetTimerValue(timer);
    total_time += time;
    if(DEBUG_PRINT)
    	printf("Run time %f\n\n", time);
    cutResetTimer(timer);

    cudaMemset(device_ret.data + GENOMIC_DATA_COUNT * IMAGE_MxN, 0, sizeof(int) * IMAGE_MxN);
    cudaMemset(device_ret.data, 0, GENOMIC_DATA_COUNT * sizeof(int) * IMAGE_MxN);
    buildImage(device_ret.data + GENOMIC_DATA_COUNT * IMAGE_MxN,
    											device_labels.data,device_indices.data);


    if (RUN_DISPLAY){
        expandConstantImage(device_regular_pbo, device_ret.data + GENOMIC_DATA_COUNT * IMAGE_MxN);
        //    for (int i=0; i<GENOMIC_DATA_COUNT; i++)
        //    	buildSplitImage<<<grid,block>>>(device_ret.data + i * IMAGE_MxN,device_labels.data,device_indices.data,i);
        //
        //	expandLogImage<<<grid,block>>>(device_log_pbo, device_ww_count.data + GENOMIC_DATA_COUNT * IMAGE_MxN);
        //	generateSplitImage(genome_index, device_split_pbo);
    }
    else{
    	std::ofstream file;
    	std::stringstream tmp;
    	tmp << CONFIG_PATH << "indices" << counter;
		std::string filename(tmp.str());

		MATRIX<unsigned int> save_labels(device_labels.row,device_labels.col);
		cutilSafeCall(cudaMemcpy(save_labels.data, device_labels.data, sizeof(unsigned int) * save_labels.row * save_labels.col, cudaMemcpyDeviceToHost));
		MATRIX<unsigned int> indices(device_indices.row, device_indices.col);
		cutilSafeCall(cudaMemcpy(indices.data, device_indices.data, sizeof(unsigned int) * indices.row * indices.col, cudaMemcpyDeviceToHost));

    	file.open(filename.c_str());

    	for (int i=0; i<DATA_SIZE; i++){
    		file << indices.data[i] << " " << save_labels.data[i] << std::endl;
    	}

		//wtf?
		delete save_labels.data;
		save_labels.data= 0;
		delete indices.data;
		indices.data= 0;

    	file.close();
    }

	if (DEBUG_PRINT)
    	printf("Total Time: %f\n\n", total_time);

   	return EXIT_SUCCESS;
}

int SOM::make_data(int n,int S, int F,float weight,
		MATRIX<MATRIX_TYPE> &pc1,
		MATRIX<MATRIX_TYPE> &pc2,
		ORDERED_MATRIX<MATRIX_TYPE, ROW_MAJOR> &x)
{
	float center_vec[F];
	for (int i=0; i<S; i++){
		for (int cv_f = 0; cv_f < F; cv_f++){
			center_vec[cv_f] = (float)rand() / (float)RAND_MAX - 0.5;
		}
		for (int j=0; j<n; j++){
			for (int cv_f = 0; cv_f < F; cv_f++){
				x((i * n + j), cv_f) = weight * center_vec[cv_f] + (float)rand()/ (float)RAND_MAX - 0.5;
			}
		}
	}

}
