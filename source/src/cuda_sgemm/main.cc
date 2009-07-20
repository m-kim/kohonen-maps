#include "shared.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#ifdef __APPLE__
#include <GL/glew.h>
#include <GLUT/glut.h>
#elif defined(_WIN32)
#include <windows.h>
#include "GL/glut.h"
#else
#include <cstdlib>
#include <GL/glew.h>
#include <GL/glut.h>
#endif

#include <cmath>
#include <iostream>
#include <fstream>


#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include <sstream>

#include "Tokenizer.h"
#include "SOM.h"

SOM som;
#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB

int split_som_window = 0, som_window = 0;

float bin1, bin2;
uint *d_split_output;
uchar4 *d_regular_output;
unsigned char *d_log_output;

GLuint split_pbo = 0, log_pbo = 0;          // OpenGL pixel buffer object
GLuint pbo = 0;

GLuint displayRegTex = 0, displaySplitTex = 0, display_log_tex = 0;
unsigned int width = 1024, height = 1024;


// display results using OpenGL (called by GLUT)
void display()
{
	glutSetWindow(som_window);

	if (som.RUN_CYCLE && som.counter < som.host_T){
		som.counter++;
		som.updateWeights();
		som.runCuda((uint*)d_regular_output, d_split_output, d_log_output);
		som.updateConvergence();
	}

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// download image from PBO to OpenGL texture
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture  (GL_TEXTURE_TYPE, displayRegTex);
	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_TYPE,
							0, 0, 0, width/2, height/2, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutReportErrors();
	glEnable(GL_TEXTURE_TYPE);

	// draw textured quad
	glDisable(GL_DEPTH_TEST);
	glBegin(GL_QUADS);
	glTexCoord2f(0    , height/2);  glVertex2f(0, 0);
	glTexCoord2f(width/2, height/2);  glVertex2f(1, 0);
	glTexCoord2f(width/2, 0     );  glVertex2f(1, 1);
	glTexCoord2f(0    , 0     );  glVertex2f(0, 1);
	glEnd();
	glDisable(GL_TEXTURE_TYPE);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, split_pbo);
	glBindTexture  (GL_TEXTURE_TYPE, displaySplitTex);
	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_TYPE,
					0, 0, 0, width/2, height/2, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glEnable(GL_TEXTURE_TYPE);

	// draw textured quad
	glBegin(GL_QUADS);
	glTexCoord2f(0    , height/2);  glVertex2f(1, 0);
	glTexCoord2f(width/2, height/2);  glVertex2f(2, 0);
	glTexCoord2f(width/2, 0     );  glVertex2f(2, 1);
	glTexCoord2f(0    , 0     );  glVertex2f(1, 1);
	glEnd();
	glDisable(GL_TEXTURE_TYPE);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, log_pbo);
	glBindTexture  (GL_TEXTURE_TYPE, display_log_tex);
	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_TYPE,
					0, 0, 0, width/2, height/2, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);
	glEnable(GL_TEXTURE_TYPE);

	// draw textured quad
	glBegin(GL_QUADS);
	glTexCoord2f(0    , height/2);  glVertex2f(0, 1);
	glTexCoord2f(width/2, height/2);  glVertex2f(1, 1);
	glTexCoord2f(width/2, 0     );  glVertex2f(1, 2);
	glTexCoord2f(0    , 0     );  glVertex2f(0, 2);
	glEnd();
	glDisable(GL_TEXTURE_TYPE);

    glutSwapBuffers();
//    glutReportErrors();

}


void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case 'n':
    case 'N':
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );

        som.updateWeights();
        som.runCuda((uint*)d_regular_output, d_split_output, d_log_output);
        som.updateConvergence();
       	cutilSafeCall(cudaGLUnmapBufferObject(pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(split_pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(log_pbo) );

    	break;
    case 'r':
    case 'R':
    	som.genome_index = ++som.genome_index % GENOMIC_DATA_COUNT;
    	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
    	som.generateSplitImage(som.genome_index, d_split_output);
       	cutilSafeCall(cudaGLUnmapBufferObject(split_pbo) );

       	printf("Genome data %d\n", som.genome_index + 1);
    	break;
	case 27:
		exit(0);
		break;

	default:
		break;
    }
    glutPostRedisplay();
}

void initPBO()
{
    if (pbo) {
        // delete old buffer
        cutilSafeCall(cudaGLUnregisterBufferObject(pbo));
        glDeleteBuffersARB(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(uchar4), 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    cutilSafeCall(cudaGLRegisterBufferObject(pbo));

    // create texture for display
    if (displayRegTex) {
        glDeleteTextures(1, &displayRegTex);
    }
    glGenTextures(1, &displayRegTex);
    glBindTexture  (GL_TEXTURE_TYPE, displayRegTex);
    glTexImage2D   (GL_TEXTURE_TYPE, 0, GL_RGBA8, width/2, height/2, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture  (GL_TEXTURE_TYPE, 0);

}
void initSplitPBO()
{
    if (split_pbo) {
        // delete old buffer
        cutilSafeCall(cudaGLUnregisterBufferObject(split_pbo));
        glDeleteBuffersARB(1, &split_pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &split_pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, split_pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(uint), 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    cutilSafeCall(cudaGLRegisterBufferObject(split_pbo));

    // create texture for display
    if (displaySplitTex) {
        glDeleteTextures(1, &displaySplitTex);
    }
    glGenTextures(1, &displaySplitTex);
    glBindTexture  (GL_TEXTURE_TYPE, displaySplitTex);
    glTexImage2D   (GL_TEXTURE_TYPE, 0, GL_RGBA8, width/2, height/2, 0, GL_RGBA, GL_UNSIGNED_INT, NULL);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture  (GL_TEXTURE_TYPE, 0);

}
void initLogPBO()
{
    if (log_pbo) {
        // delete old buffer
        cutilSafeCall(cudaGLUnregisterBufferObject(log_pbo));
        glDeleteBuffersARB(1, &log_pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &log_pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, log_pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(unsigned char), 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    cutilSafeCall(cudaGLRegisterBufferObject(log_pbo));

    // create texture for display
    if (display_log_tex) {
        glDeleteTextures(1, &display_log_tex);
    }
    glGenTextures(1, &display_log_tex);
    glBindTexture  (GL_TEXTURE_TYPE, display_log_tex);
    glTexImage2D   (GL_TEXTURE_TYPE, 0, GL_LUMINANCE, width/2, height/2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture  (GL_TEXTURE_TYPE, 0);

}

void initGLBuffers()
{
	initPBO();
	initSplitPBO();
	initLogPBO();
}
void reshape(int x, int y)
{
    width = x; height = y;

    //initGLBuffers();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 2.0, 0.0, 2.0, 0.0, 1.0);
}
void idle()
{
    glutPostRedisplay();
}

void initGL( int argc, char **argv )
{
    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    som_window = glutCreateWindow("CUDA SOM");

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
//    glutMouseFunc(mouse);
//    glutMotionFunc(motion);
    glutReshapeFunc(reshape);


    glutIdleFunc(idle);


#ifndef __APPLE__
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "
                         "GL_ARB_pixel_buffer_object "
                         ))
    {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(-1);
    }
#endif

}
void readConfig()
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
int getLineCount(std::string name)
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
		line_count++;
	file.close();
	return line_count;
}
int getFile(std::string name, ORDERED_MATRIX<MATRIX_TYPE,  HOST, ROW_MAJOR> &x, uint *labels, int offset, uint label_value)
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
		line_count++;
	file.clear();
	file.seekg(0);

	while (file.good()){
		getline(file, str);
		if (isdigit(str.c_str()[0])){
			char *tok = strtok((char*)str.c_str(), ",");
			for (int i=0; i<som.VECTOR_SIZE; i++){
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
	readConfig();
	if (som.RUN_DISPLAY)
		initGL(argc,argv);


	ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> x;
	uint *labels = 0;

	std::ifstream file;
	char filename[100];
	std::string str;
	int row = 0;
	int line_count = 0;
	line_count += getLineCount("anoGAm1-100k.fa");
	line_count += getLineCount("cb3-100k.fa");
	line_count += getLineCount("ce2-100k.fa");
	line_count += getLineCount("dm2-100k.fa");
	line_count += getLineCount("dp3-100k.fa");
	line_count += getLineCount("galgal2-100k.fa");
	line_count += getLineCount("fr2-100k.fa");
	line_count += getLineCount("tetnig1-100k.fa");
	line_count += getLineCount("ci1-100k.fa");
//	line_count += getLineCount("danrer3-100k.fa");

	x.row = line_count;
	x.col = som.VECTOR_SIZE;
	x.data = (MATRIX_TYPE*)malloc(sizeof(MATRIX_TYPE) * x.row * x.col);	labels = (uint*)malloc(sizeof(uint) * x.row);
	som.DATA_SIZE = line_count;

	row += getFile("anoGAm1-100k.fa", x, labels, 0, 0);
	row += getFile("cb3-100k.fa", x, labels, row, 1);
	row += getFile("ce2-100k.fa", x, labels, row, 2);
	row += getFile("dm2-100k.fa", x, labels, row, 3);
	row += getFile("dp3-100k.fa", x, labels, row, 4);
	row += getFile("galgal2-100k.fa", x, labels, row, 5);
	row += getFile("fr2-100k.fa", x, labels, row, 6);
	row += getFile("tetnig1-100k.fa", x, labels, row, 7);
	row += getFile("ci1-100k.fa", x, labels, row, 8);
//	row += getFile("danrer3-100k.fa", x, labels, row, 9);
	//getFile("hg17-100k.fa", x, labels, 2627 + 956 + 999 + 1052 + 1339 + 8236, 6);


//	getFile(som.DATA_FILE, x, labels, 0,0);

//	make_data(2000, GENOMIC_DATA_COUNT, VECTOR_SIZE, 3.0, pc1, pc2, x);
//	for (int i=0; i<GENOMIC_DATA_COUNT; i++){
//		for (int j=0; j<2000; j++){
//			labels[i * 2000 + j] = i;
//		}
//	}
	MATRIX<float, HOST> pc1(som.VECTOR_SIZE, 1);

	MATRIX<float, HOST> pc2(som.VECTOR_SIZE, 1);


	//x.normalize();
	x.pca(pc1,pc2);

	if (som.DEBUG_PRINT){
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

	ORDERED_MATRIX<float, HOST, COLUMN_MAJOR> data_dm(som.DATA_SIZE, som.VECTOR_SIZE);
	MATRIX<float, HOST> pd1(som.DATA_SIZE, 1);
	MATRIX<float, HOST> pd2(som.DATA_SIZE, 1);

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

	bin1 = 2 * som.EXPANSION * std1 / IMAGE_N;
	bin2 = 2 * som.EXPANSION * std2 / IMAGE_M;

	if (som.DEBUG_PRINT){
		printf("Std dev: %f %f\n", std1,std2);
		printf("scale %f %f %d %d\n", bin1, bin2, IMAGE_M, IMAGE_N);
	}

/*************************************************************************************
 * init_ww and
 * musical_chairs
 */
	float *b1 = (float*)malloc(sizeof(float) * som.VECTOR_SIZE);
	float *b2 = (float*)malloc(sizeof(float) * som.VECTOR_SIZE);
	for (int i=0; i<pc1.row; i++){
		b1[i] = pc1.data[i] * bin1;
		b2[i] = pc2.data[i] * bin2;
	}

	ORDERED_MATRIX<float, HOST, COLUMN_MAJOR> ww(som.VECTOR_SIZE, IMAGE_MxN);

	//remember, mean0 = dm
	for (int i=0; i<IMAGE_N; i++){
		for (int j=0; j<IMAGE_M; j++){
			for (int k=0; k<som.VECTOR_SIZE; k++){
				ww(k,i + IMAGE_M * j) = dm[k] + (b1[k] * (i - IMAGE_N/2) + b2[k] * (j-IMAGE_M/2));
			}
		}
	}

	if(som.RUN_DISPLAY){
		// map PBO to get CUDA device pointer
		initGLBuffers();
		cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
		cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
		cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );
	}
	else{
		cutilSafeCall(cudaMalloc((void**)&d_regular_output, sizeof(unsigned int) * 512 * 512));
		cutilSafeCall(cudaMalloc((void**)&d_split_output, sizeof(unsigned int) * 512 * 512));
		cutilSafeCall(cudaMalloc((void**)&d_log_output, sizeof(unsigned int) * 512 * 512));
	}

	unsigned int timer;
    cutCreateTimer(&timer);
    double time;
    cutResetTimer(timer);
    cutStartTimer(timer);
    som.setupCuda(ww,x,labels,(uint*)d_regular_output, d_split_output, d_log_output);
	cutStopTimer(timer);
	time = cutGetTimerValue(timer);
	if (som.DEBUG_PRINT)
		printf("Setup time %f\n\n", time);

	som.runCuda((uint*)d_regular_output, d_split_output, d_log_output);

	if (som.RUN_DISPLAY){
		cutilSafeCall( cudaGLUnmapBufferObject(split_pbo) );
		cutilSafeCall( cudaGLUnmapBufferObject(log_pbo) );
		cutilSafeCall( cudaGLUnmapBufferObject(pbo) );
	}

	if (som.RUN_DISPLAY)
		glutMainLoop();
	else{
		while( som.counter < som.host_T){
			som.counter++;
			som.updateWeights();
			som.runCuda((uint*)d_regular_output, d_split_output, d_log_output);
			som.updateConvergence();
		}
		cutilSafeCall(cudaFree(d_regular_output));
		cutilSafeCall(cudaFree(d_split_output));
		cutilSafeCall(cudaFree(d_log_output));
	}

	delete dm, b1,b2,labels;
};
