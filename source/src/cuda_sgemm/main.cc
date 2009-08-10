#include "shared.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#define GL_GLEXT_PROTOTYPES

#ifdef __APPLE__
#elif defined(_WIN32)
#include <windows.h>
#else
#endif
#include <GL/gl.h>
#include <GL/glext.h>

#include <cmath>
#include <iostream>
#include <fstream>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include <sstream>

#include "Tokenizer.h"
#include "SOM.h"
#include "HistSOMWidget.h"
#include <qapplication.h>
#include "window.h"

HistSOMWidget *widget;


#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB

int split_som_window = 0, som_window = 0;

float bin1, bin2;

//GLuint split_pbo = 0, log_pbo = 0;          // OpenGL pixel buffer object
//GLuint pbo = 0;

//GLuint displayRegTex = 0, displaySplitTex = 0, display_log_tex = 0;
unsigned int width = 1024, height = 1024;



//void keyboard(unsigned char key, int /*x*/, int /*y*/)
//{
//    switch(key) {
//    case '+':
//		som.increaseLuminance();
//    	break;
//    case '-':
//    	som.decreaseLuminance();
//    case 'n':
//    case 'N':
//       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
//       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
//       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );
//
//        som.updateWeights();
//        som.runCuda();
//        som.updateConvergence();
//       	cutilSafeCall(cudaGLUnmapBufferObject(pbo) );
//       	cutilSafeCall(cudaGLUnmapBufferObject(split_pbo) );
//       	cutilSafeCall(cudaGLUnmapBufferObject(log_pbo) );
//
//    	break;
//    case 'r':
//    case 'R':
//    	som.genome_index = ++som.genome_index % GENOMIC_DATA_COUNT;
//    	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
//    	som.generateSplitImage(som.genome_index, d_split_output);
//       	cutilSafeCall(cudaGLUnmapBufferObject(split_pbo) );
//
//       	printf("Genome data %d\n", som.genome_index + 1);
//    	break;
//	case 27:
//		exit(0);
//		break;
//
//	default:
//		break;
//    }
//
//    glutPostRedisplay();
//}


void readConfig(SOM &som)
{
	std::ifstream file;
	char filename[100];
	sprintf(filename, "%s%s",CONFIG_PATH,"config.txt");
	file.open(filename, std::ifstream::in);
	std::string str;
	if (!file.good()){
		printf("Configuration file missing!\n");
		exit(-1);
	}

	while (file.good()){
		getline(file, str);

		Tokenizer tok(str);
		if (str.find("BETA") != std::string::npos){
			som.BETA = atoi(tok(1).c_str());
		}
		else if (str.find("ALPHA") != std::string::npos){
			som.ALPHA = atof(tok(1).c_str());
		}
		else if (str.find("ITERATIONS") != std::string::npos){
			som.host_T = atoi(tok(1).c_str());
		}
		else if (str.find("DEBUG_PRINT") != std::string::npos){
			som.DEBUG_PRINT = atoi(tok(1).c_str());
		}
		else if (str.find("RUN_CYCLE") != std::string::npos){
			som.RUN_CYCLE = atoi(tok(1).c_str());
		}
		else if (str.find("RUN_DISPLAY") != std::string::npos){
			som.RUN_DISPLAY = atoi(tok(1).c_str());
		}
		else if (str.find("SAVE_FILES") != std::string::npos){
			som.SAVE_FILES = atoi(tok(1).c_str());
		}
		else if (str.find("DATA_PATH") != std::string::npos){
			som.DATA_PATH = tok(1);
		}
		else if (str.find("DATA_FILE") != std::string::npos){
			som.DATA_FILE = tok(1);
		}
		else if (str.find("VECTOR_SIZE") != std::string::npos){
			som.VECTOR_SIZE = atoi(tok(1).c_str());
		}
		else if (str.find("EXPANSION") != std::string::npos){
			som.EXPANSION = atoi(tok(1).c_str());
		}
	}
}
int getLineCount(std::string name, SOM &som)
{
	std::ifstream file;
	std::stringstream filename;
	filename << som.DATA_PATH << name;
	file.open(filename.str().c_str(), std::ifstream::in);
	std::string str;

	int line_count = 0;
	int row = 0;
	if (!file.good()){
		std::cout << "Data file missing!  " << filename.str() << std::endl;
		exit(-1);
	}
	while (getline(file,str))
		if (isdigit(str.c_str()[0])){
			line_count++;
		}
	file.close();
	return line_count;
}
int getFile(std::string name, ORDERED_MATRIX<MATRIX_TYPE,  HOST, ROW_MAJOR> &x, uint *labels, int offset, uint label_value)
{
	std::ifstream file;
	std::stringstream filename;
	filename << widget->som.DATA_PATH << name;
	file.open(filename.str().c_str(), std::ifstream::in);
	std::string str;

	int line_count = 0;
	int row = 0;
	if (!file.good()){
		std::cout << "Data file missing!  " << filename.str() << std::endl;
		exit(-1);
	}
	while (getline(file,str))
		line_count++;
	file.clear();
	file.seekg(0);

	while (file.good()){
		getline(file, str);
		if (isdigit(str.c_str()[0])){
			char *tok = strtok((char*)str.c_str(), ",");
			for (int i=0; i<widget->som.VECTOR_SIZE; i++){
				//16 rows by 20000 cols in the file
				x(offset + row, i) = atof(tok);
				tok = strtok(NULL, " ");
			}
			labels[offset + row] = label_value;
			row++;
		}
	}


	file.close();
	return row;
}
//int getFile(std::string name, ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> &x, uint *&labels, uint offset, uint label_value)
//{
//	std::ifstream file;
//	std::stringstream filename;
//	filename << som.DATA_PATH << name;
//	file.open(filename.str().c_str(), std::ifstream::in);
//	std::string str;
//
//	int line_count = 0;
//	int row = 0;
//	if (!file.good()){
//		std::cout << "Data file missing!  " << filename.str() << std::endl;
//		exit(-1);
//	}
//	while (getline(file,str))
//		line_count++;
//	file.clear();
//	file.seekg(0);
//
//	x.row = line_count;
//	x.col = som.VECTOR_SIZE;
//	x.data = (MATRIX_TYPE*)malloc(sizeof(MATRIX_TYPE) * x.row * x.col);
//	labels = (uint*)malloc(sizeof(uint) * x.row);
//
//	som.DATA_SIZE = line_count;
//	while (getline(file, str)){
//		//if (isdigit(str.c_str()[0])){
//		char *tok = strtok((char*)str.c_str(), " ");
//
//		x(row, 0) = atof(tok);
//		for (int i=1; i<som.VECTOR_SIZE; i++){
//			tok = strtok(NULL, " ");
//			x(row, i) = atof(tok);
//		}
//		tok = strtok(NULL, " ");
//		labels[row] = atoi(tok);
//		row++;
//	}
//
//	if (som.DEBUG_PRINT)
//		printf("row: %d %d\n",row, line_count);
//	file.close();
//	return row;
//}

int main( int argc, char **argv )
{
	QApplication a(argc,argv);

	widget = new HistSOMWidget();
	readConfig(widget->som);

//	if (widget->som.RUN_DISPLAY)
//		initGL(argc,argv);


	ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> x;
	uint *labels = 0;

	std::ifstream file;
	char filename[100];
	std::string str;
	int line_count = 0;
	line_count += getLineCount("anoGAm1-100k.fa",widget->som);
	line_count += getLineCount("cb3-100k.fa",widget->som);
	line_count += getLineCount("ce2-100k.fa",widget->som);
	line_count += getLineCount("dm2-100k.fa",widget->som);
	line_count += getLineCount("dp3-100k.fa",widget->som);
	line_count += getLineCount("galgal2-100k.fa",widget->som);
	line_count += getLineCount("fr2-100k.fa",widget->som);
	line_count += getLineCount("tetnig1-100k.fa",widget->som);
	line_count += getLineCount("ci1-100k.fa",widget->som);
//	line_count += getLineCount("danrer3-100k.fa");

	x.row = line_count;
	x.col = widget->som.VECTOR_SIZE;
	x.data = (MATRIX_TYPE*)malloc(sizeof(MATRIX_TYPE) * x.row * x.col);	labels = (uint*)malloc(sizeof(uint) * x.row);
	widget->som.DATA_SIZE = line_count;

	int row = 0;
	row += getFile("anoGAm1-100k.fa", x, labels, 0, 0);
	row += getFile("cb3-100k.fa", x, labels, row, 1);
	row += getFile("ce2-100k.fa", x, labels, row, 2);
	row += getFile("dm2-100k.fa", x, labels, row, 3);
	row += getFile("dp3-100k.fa", x, labels, row, 4);
	row += getFile("galgal2-100k.fa", x, labels, row, 5);
	row += getFile("fr2-100k.fa", x, labels, row, 6);
	row += getFile("tetnig1-100k.fa", x, labels, row, 7);
	row += getFile("ci1-100k.fa", x, labels, row, 8);

	if (widget->som.DEBUG_PRINT)
		printf("%d %d\n", row, line_count);
//	row += getFile("danrer3-100k.fa", x, labels, row, 9);
	//getFile("hg17-100k.fa", x, labels, 2627 + 956 + 999 + 1052 + 1339 + 8236, 6);

//	getFile(som.DATA_FILE, x, labels, 0,0);

//	make_data(2000, GENOMIC_DATA_COUNT, VECTOR_SIZE, 3.0, pc1, pc2, x);
//	for (int i=0; i<GENOMIC_DATA_COUNT; i++){
//		for (int j=0; j<2000; j++){
//			labels[i * 2000 + j] = i;
//		}
//	}
	MATRIX<float, HOST> pc1(widget->som.VECTOR_SIZE, 1);
	MATRIX<float, HOST> pc2(widget->som.VECTOR_SIZE, 1);

	//x.normalize();
	x.pca(pc1,pc2);

	if (widget->som.DEBUG_PRINT){
		printf("pc1: ");
		pc1.print();
		printf("pc2: ");
		pc2.print();
	}
	//mean0 == dm
	//dm should be shape = (16,)
	//dm is correct compared to python code...
	//it needed a reverse index
	float *dm = x.mean();//(float*)malloc(sizeof(float) * x.col);

	ORDERED_MATRIX<float, HOST, COLUMN_MAJOR> data_dm(widget->som.DATA_SIZE, widget->som.VECTOR_SIZE);
	MATRIX<float, HOST> pd1(widget->som.DATA_SIZE, 1);
	MATRIX<float, HOST> pd2(widget->som.DATA_SIZE, 1);

	for (int i=0; i<data_dm.row; i++){
		for (int j=0; j<data_dm.col; j++){
			data_dm(i,j) = x(i,j) - dm[j];
		}
		pd1.data[i] = pc1.dot(data_dm,i);
		pd2.data[i] = pc2.dot(data_dm,i);
	}

/*****************************************************************************************
 * scale map
 */
	float std1 = pd1.stdDev();
	float std2 = pd2.stdDev();

	//resize...
	//ummm...this is only if M is "None"
	//M = int(N * std2 / std1);

	bin1 = 2 * widget->som.EXPANSION * std1 / IMAGE_Y;
	bin2 = 2 * widget->som.EXPANSION * std2 / IMAGE_X;

	if (widget->som.DEBUG_PRINT){
		printf("Std dev: %f %f\n", std1,std2);
		printf("scale %f %f %d %d\n", bin1, bin2, IMAGE_X, IMAGE_Y);
	}

/*************************************************************************************
 * init_ww and
 * musical_chairs
 */
	float *b1 = (float*)malloc(sizeof(float) * widget->som.VECTOR_SIZE);
	float *b2 = (float*)malloc(sizeof(float) * widget->som.VECTOR_SIZE);
	for (int i=0; i<pc1.row; i++){
		b1[i] = pc1.data[i] * bin1;
		b2[i] = pc2.data[i] * bin2;
	}

	ORDERED_MATRIX<float, HOST, COLUMN_MAJOR> ww(widget->som.VECTOR_SIZE, IMAGE_XxY);

	//remember, mean0 = dm
	for (int i=0; i<IMAGE_Y; i++){
		for (int j=0; j<IMAGE_X; j++){
			for (int k=0; k<widget->som.VECTOR_SIZE; k++){
				ww(k,i * IMAGE_X + j) = dm[k] + (b1[k] * (i - IMAGE_Y/2) + b2[k] * (j-IMAGE_X/2));
			}
		}
	}

//	if(widget->som.RUN_DISPLAY){
//		// map PBO to get CUDA device pointer
//		initGLBuffers();
//		cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
//		cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
//		cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );
//	}
//	else{
//		cutilSafeCall(cudaMalloc((void**)&d_regular_output, sizeof(unsigned int) * 512 * 512));
//		cutilSafeCall(cudaMalloc((void**)&d_split_output, sizeof(unsigned int) * 512 * 512));
//		cutilSafeCall(cudaMalloc((void**)&d_log_output, sizeof(unsigned int) * 512 * 512));
//	}
	widget->show();

	unsigned int timer;
    cutCreateTimer(&timer);
    double time;
    cutResetTimer(timer);
    cutStartTimer(timer);
    widget->setupCuda(ww,x,labels);
	cutStopTimer(timer);
	time = cutGetTimerValue(timer);
	if (widget->som.DEBUG_PRINT)
		printf("Copy everything to memory: %f ms\n\n", time);

	widget->som.runCuda();

	widget->unMap();
	Window window(widget);
	window.show();

//	reshape(width,height);
//	if (som.RUN_DISPLAY)
//		glutMainLoop();
//	else{
//		while( som.counter < som.host_T){
//			som.counter++;
//			som.updateWeights();
//			som.runCuda();
//			som.updateConvergence();
//		}
//		cutilSafeCall(cudaFree(d_regular_output));
//		cutilSafeCall(cudaFree(d_split_output));
//		cutilSafeCall(cudaFree(d_log_output));
//	}

	delete dm, b1,b2,labels;
	return a.exec();
};
