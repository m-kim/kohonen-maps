#ifndef SOM_H
#define SOM_H
#include "Matrix.h"
#include "shared.h"
class SOM
{
public:
	SOM();
	void setupCuda(ORDERED_MATRIX<MATRIX_TYPE, COLUMN_MAJOR> &ww,
			ORDERED_MATRIX<MATRIX_TYPE, ROW_MAJOR> &data,
			uint *labels,
			unsigned int *device_regular_pbo,
			uint *device_split_pbo,
			unsigned char *device_log_pbo);
	int runCuda(unsigned int *device_regular_pbo,
			unsigned int *device_split_pbo,
			unsigned char *device_log_pbo);

	void updateConvergence();

	void generateSplitImage(int g_index, unsigned int * device_split_pbo);

	void updateWeights();

	int make_data(int n,int S, int F,float weight,
			MATRIX<MATRIX_TYPE> &pc1,
			MATRIX<MATRIX_TYPE> &pc2,
			ORDERED_MATRIX<MATRIX_TYPE, ROW_MAJOR> &x);

	MATRIX<MATRIX_TYPE> device_ww2, device_save, device_sum, device_scratch;
	MATRIX<unsigned int> device_labels, device_indices,device_ww_count, device_ret,device_ww_count2;
	ORDERED_MATRIX<MATRIX_TYPE, COLUMN_MAJOR> device_ww, device_data;

	int genome_index;
	int DATA_SIZE;
	int VECTOR_SIZE;
	int BETA;
	float ALPHA;
	int host_T;
	int DEBUG_PRINT;
	int RUN_CYCLE;
	float host_alpha[2];
	int host_r, host_beta[2];
	int RUN_DISPLAY;

	int counter;
};

#endif
