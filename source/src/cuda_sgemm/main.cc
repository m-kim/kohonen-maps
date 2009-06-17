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
extern "C" void setupCuda(MATRIXf ww,  MATRIXf data, uint *labels, unsigned int *device_pbo);
extern "C" int runCuda(unsigned int *device_pbo);
extern "C" void cleanup();
extern "C" void sgesvd_(const char* jobu, const char* jobvt, const int* M, const int* N,
        float* A, const int* lda, float* S, float* U, const int* ldu,
        float* VT, const int* ldvt, float* work,const int* lwork, const
        int* info);

int expansion = 4;
float bin1, bin2;
uchar4 *d_output;

GLuint pbo        = 0;          // OpenGL pixel buffer object
GLuint displayTex = 0;
unsigned int width = 512, height = 512;

float stdDev(const MATRIXf &mat)
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
	return sqrt(sum / mat.row);
}

float dot(MATRIXf one, MATRIXf two, int col)
{
	float sum = 0;
	for (int i=0; i<one.row; i++){
		sum += one.data[i] * two.data[i * two.col + col ];
	}
	return sum;
}

float * mean(const MATRIXf dd)
{
	//get the mean value for each row...
	float *ret = (float*)malloc(sizeof(float) * dd.row);

	for(int i=0; i<dd.row;i++){
		ret[i] = 0.0;
		for(int j=0; j<dd.col; j++){
			ret[i] += dd.data[i * dd.col + j];
		}
		ret[i] = ret[i] / dd.col;
	}
	return ret;
}
void cov(const MATRIXf dd, float *covariance)
{
	float *mean_val = mean(dd);
	for(int i=0; i<dd.row; i++){
		for (int j=0; j<dd.row;j++){
			covariance[i * dd.row + j] = 0;
			for (int k=0; k < dd.col; k++){
				covariance[i * dd.row + j] += (dd.data[i * dd.col + k] - mean_val[i]) * (dd.data[j * dd.col + k] - mean_val[j]);
			}
			covariance[i * dd.row + j] /= (dd.col -  1);
//#if DEBUG_PRINT
			printf("%f ", covariance[i * dd.row + j]);
//#endif
		}
//#if DEBUG_PRINT
		printf("\n");
//#endif
	}

	delete mean_val;
}

//column major order doesn't matter for sgesvd_
//the matrix pumped out of cov is singular
//however, the matrix returned will be in column major order
//so, that needs to be taken into account...
void pca(MATRIXf x, MATRIXf pca1, MATRIXf pca2)
{
	//16 x 20000 means a 16x16 covariance matrix
	float *cov_mat = (float*)malloc(sizeof(float) * x.row * x.row);

	cov(x,cov_mat);
	char JOBU = 'A';
	char JOBV = 'A';

	int svd_M = x.row;
	int svd_N = x.row;
	int lda = x.row;
	int ldu = x.row;
	int ldv = x.row;
	int info;
	int lwork;
	float s[x.row];
	float uu[x.row*x.row];
	float vv[x.row*x.row];
	float wk[201];

	sgesvd_(&JOBU, &JOBV,
			&svd_N, &svd_M,
			cov_mat, &lda,
			s,
			uu, &ldu,
			vv, &ldv,
			wk, &lwork,
			&info);


	for (int i=0; i<x.row; i++){
		pca1.data[i] = vv[i];
		pca2.data[1] = vv[i + x.row];
	}
//	memcpy(pca1, &vv[0], sizeof(float)* mat_m);
//	memcpy(pca2, &vv[1], sizeof(float)* mat_m);

#if DEBUG_PRINT
	for (int i=0; i<x.row; i++){
		printf("\t%f ", s[i]);
		for (int j=0; j<x.row; j++)
			printf("%f ", vv[IDX2C(i,j, x.row)]);
		printf("\n");
	}
	printf("\n");
	printf("%d\n",info);
#endif
	delete cov_mat;
}

//normalizes along the column
//ie if its a 16 x 20000 matrix, it normalizes the 16 vector
static void normalize(MATRIXf mat)
{
	float sum = 0;
	for (int i=0;i<mat.col; i++){
		sum = 0;
		for (int j=0; j<mat.row; j++){
			mat.data[i * mat.row + j] = fabs(mat.data[i * mat.row + j]);
			sum += mat.data[i * mat.row + j];
		}
		for (int j=0;j<mat.row; j++){
			mat.data[i * mat.row + j] /= sum;
		}
	}
}

int make_data(int n,int S, int F,float weight, MATRIXf pc1, MATRIXf pc2, MATRIXf x)
{

	float center_vec[F];
	for (int i=0; i<S; i++){
		for (int cv_f = 0; cv_f < F; cv_f++){
			center_vec[cv_f] = (float)rand() / (float)RAND_MAX - 0.5;
		}
		for (int j=0; j<n; j++){
			for (int cv_f = 0; cv_f < F; cv_f++){
				x.data[(i * n + j) * F + cv_f] = weight * center_vec[cv_f] + (float)rand()/ (float)RAND_MAX - 0.5;
				printf("%f ", x.data[(i * n + j) * F + cv_f]);
			}
			printf("\n");
		}
	}

}

// display results using OpenGL (called by GLUT)
void display()
{

	// Common diplay path
	{
		// display results
		glClear(GL_COLOR_BUFFER_BIT);

		// download image from PBO to OpenGL texture
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture  (GL_TEXTURE_TYPE, displayTex);
		glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
		glTexSubImage2D(GL_TEXTURE_TYPE,
						0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
		glEnable(GL_TEXTURE_TYPE);

		// draw textured quad
		glDisable(GL_DEPTH_TEST);
		glBegin(GL_QUADS);
		glTexCoord2f(0    , height);  glVertex2f(0, 0);
		glTexCoord2f(width, height);  glVertex2f(1, 0);
		glTexCoord2f(width, 0     );  glVertex2f(1, 1);
		glTexCoord2f(0    , 0     );  glVertex2f(0, 1);
		glEnd();
		glDisable(GL_TEXTURE_TYPE);
		glDisable(GL_FRAGMENT_PROGRAM_ARB);

		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	}

    glutSwapBuffers();
    glutReportErrors();

}
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case 'n':
    case 'N':
        runCuda((unsigned int*)d_output);

    	break;
        case 27:
            exit(0);
            break;

        default:
            break;
    }
    glutPostRedisplay();
}
void initGLBuffers()
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
    if (displayTex) {
        glDeleteTextures(1, &displayTex);
    }
    glGenTextures(1, &displayTex);
    glBindTexture  (GL_TEXTURE_TYPE, displayTex);
    glTexImage2D   (GL_TEXTURE_TYPE, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture  (GL_TEXTURE_TYPE, 0);

    // calculate new grid size
//    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
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
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
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
    glutCreateWindow("CUDA bicubic texture filtering");
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


int main( int argc, char **argv )
{

	initGL(argc,argv);

	MATRIXf pc1;
	pc1.data = (float*)malloc(sizeof(float) * 16);
	pc1.row = 16;
	pc1.col = 1;

	MATRIXf pc2;
	pc2.data = (float*)malloc(sizeof(float) * 16);
	pc2.row = 16;
	pc2.col = 1;

	MATRIXf x;
	x.data = (float*)malloc(sizeof(float) * 20000*16);
	x.row = 16;
	x.col = 20000;

//	float dd[] = {.69, -1.31, .39, .09, 1.29,.49,.19,-.81,-.31,-.71
//				,.49, -1.21, .99, .29, 1.09, .79, -.31, -.81, -.31, -1.01
//				,1.00000,  -1.50000,  -0.39000,   0.00000,  -0.09000,   1.40000,   0.39000,  -0.90000,  -0.31000,  -0.41000
//				};
//	MATRIX mat;
//	mat.data= dd;
//	mat.row = 3;
//	mat.col = 10;
//	pca(mat, pc1, pc2);



//	make_data(1000, 20, 16, 3.0, pc1, pc2, x);
	//normalize(x);

	std::ifstream file;
	char filename[100];
	sprintf(filename, "%s/%s",SRC_PATH,"data");
	printf("%s\n",filename);
	file.open(filename, std::ifstream::in);
	std::string str;
	memset(x.data, 0, sizeof(float) * 16 * 20000);
	int row = 0;
	int col = 0;
	while (file.good()){
		//getline will retrieve 20000 numbers...
		getline(file, str);
		char *tok = strtok((char*)str.c_str(), " ");
		while (tok != NULL){
			//16 rows by 20000 cols in the file
			x.data[row * x.col + col] = atof(tok);
			tok = strtok(NULL, " ");
			col++;
		}
		col = 0;
		row++;
	}
	file.close();

	//pca(x, pc1, pc2);

	float tmp[] ={-0.36220333, -0.35487528, -0.19582513,  0.01291018,  0.25750163,  0.15277131
			 ,-0.37524103  ,0.16958821,  0.14361155, -0.2686145,   0.33782259,  0.00555586
			  ,0.29207888,  0.27095362,  0.15164405, -0.2376786};
	memcpy(pc1.data, tmp, sizeof(float) * 16);

	float tmp2[] = {0.00260173,  0.02293971, -0.47482202,  0.09716536,  0.00760736,  0.30567733
	  ,0.04289583, -0.09414034,  0.18733382, -0.01131097, -0.39201071,  0.52470409
	  ,0.29333093, -0.26513936, -0.19181659, -0.05501617};
	memcpy(pc2.data, tmp2, sizeof(float) * 16);

	//mean0 == dm
	//dm should be shape = (16,)
	//dm is correct compared to python code...
	//it needed a reverse index

	float *dm = (float*)malloc(sizeof(float) * x.row);

	for(int i=0; i<x.row;i++){
		dm[i] = 0.0;
		for(int j=0; j<x.col; j++){
			dm[i] += x.data[i * x.col + j];
		}
		dm[i] = dm[i] / x.col;
	}

	MATRIXf pd1;
	pd1.data = (float*)malloc(sizeof(float) * 20000);
	pd1.row = 20000;
	pd1.col = 1;

	MATRIXf pd2;
	pd2.data = (float*)malloc(sizeof(float) * 20000);
	pd2.row = 20000;
	pd2.col = 1;

	MATRIXf data_dm;
	data_dm.data = (float*)malloc(sizeof(float) * 20000 * 16);
	data_dm.row = 16;
	data_dm.col = 20000;

	for (int i=0; i<data_dm.col; i++){
		for (int j=0; j<data_dm.row; j++){
			data_dm.data[j * data_dm.col + i] = x.data[j * x.col + i] - dm[j];
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
	float *b1 = (float*)malloc(sizeof(float) * 16);
	float *b2 = (float*)malloc(sizeof(float) * 16);
	for (int i=0; i<pc1.row; i++){
		b1[i] = pc1.data[i] * bin1;
		b2[i] = pc2.data[i] * bin2;
	}

	MATRIXf ww;
	ww.data = (float*)malloc(sizeof(float) * IMAGE_M * IMAGE_N * 16);
	ww.row = IMAGE_M * IMAGE_N;
	ww.col = 16;


	//this doesn't work...
	//remember, mean0 = dm
	for (int i=0; i<IMAGE_N; i++){
		for (int j=0; j<IMAGE_M; j++){
			for (int k=0; k<16; k++){
				ww.data[k + (i * IMAGE_M + j) * 16 ] = dm[k] + b1[k] * (i - IMAGE_N/2) + b2[k] * (j-IMAGE_M/2);
			}
		}
	}

	uint *labels = (uint*)malloc(sizeof(uint) * x.col);
	for (int i=0; i< 20; i++){
		for (int j=0;j<1000; j++){
			labels[i * 1000 + j] = i;
		}
	}

    // map PBO to get CUDA device pointer
	initGLBuffers();

    cutilSafeCall( cudaGLMapBufferObject((void**)&d_output, pbo) );

	//chunk
	//K = 20000
	//M = 16
	//N = 896 (32 * 28)
	setupCuda(ww,x, labels,(unsigned int*)d_output);
    runCuda((unsigned int*)d_output);

	cutilSafeCall( cudaGLUnmapBufferObject(pbo) );

	glutMainLoop();


	cleanup();
	delete pc1.data, pc2.data, x, dm, pd1, pd2, data_dm, b1,b2,ww.data,labels;
};
