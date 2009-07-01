#include "shared.h"
#include <cstdlib>
#include <clapack.h>
#include <cstdio>
#include <cstring>

#include <cmath>
#include <iostream>
#include <fstream>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB
extern "C" void setupCuda(ORDERED_MATRIX<MATRIX_TYPE, COLUMN_MAJOR> ww,  ORDERED_MATRIX<MATRIX_TYPE, ROW_MAJOR> data, uint *labels, unsigned int *device_regular_pbo, uint *device_split_pbo, unsigned char *device_log_pbo);
extern "C" int runCuda(unsigned int *device_regular_pbo, unsigned int *device_split_pbo, unsigned char *device_log_pbo);
extern "C" void cleanup();

extern "C" int genome_index = 0;
extern "C" void generateSplitImage(int g_index, unsigned int * device_split_pbo);

int split_som_window = 0, som_window = 0;
float expansion = 4;
float bin1, bin2;
uint *d_split_output;
uchar4 **d_regular_output;
unsigned char *d_log_output;

GLuint split_pbo = 0, log_pbo = 0;          // OpenGL pixel buffer object
GLuint pbo = 0;

GLuint displayRegTex = 0, displaySplitTex = 0, display_log_tex = 0;
unsigned int width = 1024, height = 1024;

float stdDev(const MATRIX<MATRIX_TYPE> &mat)
{
	float mean_x = 0;
	for (int i=0; i<mat.row; i++){
		mean_x += mat.data[i];
	}
	mean_x /= mat.row;
	float sum = 0;
	for (int i=0; i<mat.row; i++){
		sum += pow(mat.data[i] - mean_x, 2);
	}
	return sqrt(sum / (mat.row - 1));
}

float dot(MATRIX<MATRIX_TYPE> one, MATRIX<MATRIX_TYPE> two, int col)
{
	float sum = 0;
	for (int i=0; i<one.row; i++){
		sum += one.data[i] * two(col,i);//.data[i * two.col + col ];
	}
	return sum;
}



//column major order doesn't matter for sgesvd_
//the matrix pumped out of cov is singular
//however, the matrix returned will be in column major order
//so, that needs to be taken into account...



//N == 10000, S == 20
int make_data(int n,int S, int F,float weight, MATRIX<MATRIX_TYPE> pc1, MATRIX<MATRIX_TYPE> pc2, MATRIX<MATRIX_TYPE> x)
{
	float center_vec[F];
	for (int i=0; i<S; i++){
		for (int cv_f = 0; cv_f < F; cv_f++){
			center_vec[cv_f] = (float)rand() / (float)RAND_MAX - 0.5;
		}
		for (int j=0; j<n; j++){
			for (int cv_f = 0; cv_f < F; cv_f++){
				x.data[(i * n + j)*F + cv_f] = weight * center_vec[cv_f] + (float)rand()/ (float)RAND_MAX - 0.5;
			}
		}
	}

}

// display results using OpenGL (called by GLUT)
void display()
{
	glutSetWindow(som_window);

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
					0, 0, 0, width/2, height/2, GL_LUMINANCE, GL_UNSIGNED_INT, 0);
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

        runCuda((uint*)d_regular_output, d_split_output, d_log_output);
       	cutilSafeCall(cudaGLUnmapBufferObject(pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(split_pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(log_pbo) );

    	break;
    case 'r':
    case 'R':
    	genome_index = ++genome_index % GENOMIC_DATA_COUNT;
    	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
    	generateSplitImage(genome_index, d_split_output);
       	cutilSafeCall(cudaGLUnmapBufferObject(split_pbo) );

       	printf("Genome data %d\n", genome_index + 1);
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
    glTexImage2D   (GL_TEXTURE_TYPE, 0, GL_LUMINANCE, width/2, height/2, 0, GL_LUMINANCE, GL_UNSIGNED_INT, NULL);
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



    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "
                         "GL_ARB_pixel_buffer_object "
                         ))
    {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(-1);
    }
}

void getFile(std::string name, ORDERED_MATRIX<MATRIX_TYPE, ROW_MAJOR> x, uint *labels, uint offset, uint label_value)
{
	std::ifstream file;
	char filename[100];
	sprintf(filename, "%s%s",DATA_PATH,name.c_str() );
	printf("%s\n", filename);
	file.open(filename, std::ifstream::in);
	std::string str;

	int counter = 0;
	int row = 0;
	if (!file.good())
		printf("file bad!\n");
	while (getline(file, str)){
		//if (isdigit(str.c_str()[0])){
		char *tok = strtok((char*)str.c_str(), " ");

		//16 rows by 20000 cols in the file
		float value1 = atof(tok);

		tok = strtok(NULL, " ");
		float value2 = atof(tok);

		x(row, 0) = cos(3.1415927 * value1 / 180.0f);
		x(row, 1) = sin(3.1415927 * value1 / 180.0f);
		x(row, 2) = cos(3.1415927 * value2 / 180.0f);
		x(row, 3) = sin(3.1415927 * value2 / 180.0f);

		if (value1 < 0 && value1 > -180){
			if (value1 < -25 && value1 > -90 && value2 > -75 && value2 < -25){
						labels[ row] = 4;
			}
			else if ( value2 > -100 && value2 < 50){
				labels[ row] = 0;
			}
			else if (value2 > 50 && value2 < 100){
				labels[ row] = 1;
			}
			else if (value2 > 50 && value2 < 180){
				counter++;
				labels[ row] = 2;
			}
			else if (value2 > -180 && value2 < -100){
				labels[ row] = 3;
			}
		}
		else if (value1 < 150 && value1 > 0 && value2 > -50 && value2 < 100){
			labels[ row] = 5;
		}
		else if (value1 < 180 && value1 > 50 && value2 > 100 && value2 < 180){
			labels[ row] = 6;
		}
		else if (value1 < 180 && value1 > 0 && value2 > -180 && value2 < 110){
			labels[ row] = 7;
		}
		else
			labels[ row] = 0;

		row++;
	}
	printf("row: %d %d\n",row, counter);
	file.close();

	printf("%d \n", counter);
}

int main( int argc, char **argv )
{

	initGL(argc,argv);

	MATRIX<float> pc1;
	pc1.data = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	pc1.row = VECTOR_SIZE;
	pc1.col = 1;

	MATRIX<float> pc2;
	pc2.data = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	pc2.row = VECTOR_SIZE;
	pc2.col = 1;

	ORDERED_MATRIX<MATRIX_TYPE, ROW_MAJOR> x;
	x.row = DATA_SIZE;
	x.col =  VECTOR_SIZE;
	x.data = (float*)malloc(sizeof(float) * x.row * x.col);

	MATRIX<float> pd1;
	pd1.row = DATA_SIZE;
	pd1.col = 1;
	pd1.data = (float*)malloc(sizeof(float) * pd1.row);

	MATRIX<float> pd2;
	pd2.row = DATA_SIZE;
	pd2.col = 1;
	pd2.data = (float*)malloc(sizeof(float) * pd2.row);

	ORDERED_MATRIX<float, COLUMN_MAJOR> data_dm;
	data_dm.row = DATA_SIZE;
	data_dm.col =  VECTOR_SIZE;
	data_dm.data = (float*)malloc(sizeof(float) * data_dm.row * data_dm.col);

	uint *labels = (uint*)malloc(sizeof(uint) * x.row);

	std::ifstream file;
	char filename[100];
	std::string str;
	int row = 0;

//	getFile("cb3.fa", x, labels, 0, 0);
//	getFile("cb3.fa", x, labels, 9581, 1);
//	getFile("ce2.fa", x, labels, 9581 + 9581, 2);
//	getFile("dm2.fa", x, labels, 9581 + 9581 + 10026, 3);

	getFile("output", x, labels, 0, 0);

//	make_data(1000, 20, VECTOR_SIZE, 3.0, pc1, pc2, x);
//	for (int i=0; i<20; i++){
//		for (int j=0; j<1000; j++){
//			labels[i * 1000 + j] = i;
//		}
//	}
	x.normalize();
	x.pca(pc1,pc2);

	printf("pc1: %f %f %f %f\n", pc1.data[0], pc1.data[1],pc1.data[2], pc1.data[3]);
	printf("pc2: %f %f %f %f\n", pc2.data[0], pc2.data[1],pc2.data[2], pc2.data[3]);

	//mean0 == dm
	//dm should be shape = (16,)
	//dm is correct compared to python code...
	//it needed a reverse index
	float *dm = x.mean();//(float*)malloc(sizeof(float) * x.col);

	for(int i=0; i<x.col;i++){
		printf("%f\n",dm[i]);
	}

	for (int i=0; i<data_dm.row; i++){
		for (int j=0; j<data_dm.col; j++){
			data_dm(i,j) = x(i,j) - dm[j];
		}
		pd1.data[i] = dot(pc1, data_dm,i);
		pd2.data[i] = dot(pc2, data_dm,i);
	}

/*****************************************************************************************
 * scale map
 */
	float std1 = stdDev(pd1);
	float std2 = stdDev(pd2);


	//resize...
	//ummm...this is only if M is "None"
	//M = int(N * std2 / std1);

	bin1 = 2 * expansion * std1 / IMAGE_N;
	bin2 = 2 * expansion * std2 / IMAGE_M;

	printf("Std dev: %f %f\n", std1,std2);
	printf("scale %f %f %d %d\n", bin1, bin2, IMAGE_M, IMAGE_N);




/*************************************************************************************
 * init_ww and
 * musical_chairs
 */
	float *b1 = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	float *b2 = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	for (int i=0; i<pc1.row; i++){
		b1[i] = pc1.data[i] * bin1;
		b2[i] = pc2.data[i] * bin2;
	}

	ORDERED_MATRIX<float, COLUMN_MAJOR> ww;
	ww.data = (float*)malloc(sizeof(float) * IMAGE_M * IMAGE_N * VECTOR_SIZE);
	ww.row = VECTOR_SIZE;
	ww.col = IMAGE_M * IMAGE_N;

	//remember, mean0 = dm
	for (int i=0; i<IMAGE_N; i++){
		for (int j=0; j<IMAGE_M; j++){
			for (int k=0; k<VECTOR_SIZE; k++){
				ww(k,i + IMAGE_M * j) = dm[k] + b1[k] * (i - IMAGE_N/2) + b2[k] * (j-IMAGE_M/2);
			}
		}
	}

    // map PBO to get CUDA device pointer
	initGLBuffers();
   	cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
   	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
   	cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );

	//chunk
	//K = 20000
	//M = 16
	//N = 896 (32 * 28)
	unsigned int timer;
    cutCreateTimer(&timer);
    double time;
    cutResetTimer(timer);
    cutStartTimer(timer);
	setupCuda(ww,x, labels,(uint*)d_regular_output, d_split_output, d_log_output);
	cutStopTimer(timer);
	time = cutGetTimerValue(timer);
	printf("Setup time %f\n\n", time);
	//for (int i=0; i<host_T; i++)
		runCuda((uint*)d_regular_output, d_split_output, d_log_output);
	cutilSafeCall( cudaGLUnmapBufferObject(split_pbo) );
	cutilSafeCall( cudaGLUnmapBufferObject(log_pbo) );
	cutilSafeCall( cudaGLUnmapBufferObject(pbo) );

	glutMainLoop();


	cleanup();
	delete pc1.data, pc2.data, x.data, dm, pd1, pd2, data_dm, b1,b2,ww.data,labels;
};
