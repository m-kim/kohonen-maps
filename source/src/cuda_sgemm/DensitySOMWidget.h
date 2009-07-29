#ifndef DENSITYSOMWIDGET_H
#define DENSITYSOMWIDGET_H
#include "SOM.h"
#include "QtSOMWidget.h"
#include "Matrix.h"

class DensitySOMWidget : public QtSOMWidget
{
public:
	DensitySOMWidget( int timerInterval=0, QWidget *parent=0, char *name=0 );
	SOM som;
	void setupCuda(ORDERED_MATRIX<MATRIX_TYPE, HOST, COLUMN_MAJOR> &ww,
			ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> &data,
			unsigned int *labels);

	void unMap()
	{
		if (som.RUN_DISPLAY){
			cutilSafeCall( cudaGLUnmapBufferObject(split_pbo) );
			cutilSafeCall( cudaGLUnmapBufferObject(log_pbo) );
			cutilSafeCall( cudaGLUnmapBufferObject(pbo) );
	       	cutilSafeCall(cudaGLUnmapBufferObject(hist_vbo) );
		}
	}
	uint *d_split_output;
	uchar4 *d_regular_output;
	unsigned int *d_log_output;
	unsigned int *d_hist_output;

protected:
	void initializeGL();
	void resizeGL( int width, int height );
	void paintGL();
	void keyPressEvent( QKeyEvent *e );
private:
	GLuint split_pbo, log_pbo;          // OpenGL pixel buffer object
	GLuint pbo;
	GLuint displayRegTex, displaySplitTex, display_log_tex;
	GLuint hist_vbo;
	int width, height;
	void initLogPBO();
	void initPBO();
	void initSplitPBO();

	void initVBO();
};
#endif
