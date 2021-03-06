#ifndef SOM_H
#define SOM_H
#include "Matrix.h"
#include "shared.h"
#include <string>

class SOM
{
public:
	SOM();
	void setupCuda(ORDERED_MATRIX<MATRIX_TYPE, HOST, COLUMN_MAJOR> &ww,
			ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> &data,
			uint *labels,
			unsigned int *device_regular_pbo,
			uint *device_split_pbo,
			unsigned int *device_log_pbo,
			unsigned int *device_hist_vbo);
	int runCuda();
	void increaseLuminance();
	void decreaseLuminance();

	void updateConvergence();

	void generateSplitImage(int g_index, unsigned int * device_split_pbo);

	void updateWeights();

	int make_data(int n,int S, int F,float weight,
			MATRIX<MATRIX_TYPE, HOST> &pc1,
			MATRIX<MATRIX_TYPE, HOST> &pc2,
			ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> &x);

	MATRIX<MATRIX_TYPE, DEVICE> device_ww2, device_sum, device_scratch, device_covariance;
	MATRIX<unsigned int, DEVICE> device_labels, device_indices,device_ww_count, device_constant_image, device_log_image,device_ww_count2, device_argmax;
	ORDERED_MATRIX<MATRIX_TYPE, DEVICE, COLUMN_MAJOR> device_ww, device_data;

	int genome_index;
	int DATA_SIZE;
	int VECTOR_SIZE;
	int BETA;
	float ALPHA;
	int host_T;
	int DEBUG_PRINT;
	int RUN_CYCLE;
	int SAVE_FILES;
	float host_alpha[2];
	int host_r, host_beta[2];
	int RUN_DISPLAY;
	std::string DATA_PATH;
	std::string DATA_FILE;
	int counter;
	int EXPANSION;

	unsigned int *device_regular_pbo;
	unsigned int *device_split_pbo;
	unsigned int *device_log_pbo;
	unsigned int *device_hist_vbo;

	double total_time;
};

#endif
