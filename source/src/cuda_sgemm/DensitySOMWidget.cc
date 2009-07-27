#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include "DensitySOMWidget.h"
#include "Matrix.h"
#include "SOM.h"

#include <QtGui/QKeyEvent>
#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB



DensitySOMWidget::DensitySOMWidget( int timerInterval, QWidget *parent, char *name):QtSOMWidget(0, parent, name)
{
	pbo = 0;
	log_pbo = 0;
	split_pbo = 0;
	displayRegTex = 0;
	displaySplitTex = 0;
	display_log_tex = 0;

	width = 1024;
	height = 1024;
}

void DensitySOMWidget::setupCuda(ORDERED_MATRIX<MATRIX_TYPE, HOST, COLUMN_MAJOR> &ww,
		ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> &data,
		unsigned int *labels)
{
	som.setupCuda(ww, data, labels,(uint *)d_regular_output, d_split_output,d_log_output);
}

void DensitySOMWidget::initLogPBO()
{
    if (log_pbo) {
        // delete old buffer
        cutilSafeCall(cudaGLUnregisterBufferObject(log_pbo));
        glDeleteBuffersARB(1, &log_pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &log_pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, log_pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(unsigned int), 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    cutilSafeCall(cudaGLRegisterBufferObject(log_pbo));

    // create texture for display
    if (display_log_tex) {
        glDeleteTextures(1, &display_log_tex);
    }
    glGenTextures(1, &display_log_tex);
    glBindTexture  (GL_TEXTURE_TYPE, display_log_tex);
    glTexImage2D   (GL_TEXTURE_TYPE, 0, GL_LUMINANCE, width/2, height/2, 0, GL_LUMINANCE, GL_UNSIGNED_INT, NULL);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture  (GL_TEXTURE_TYPE, 0);

}

void DensitySOMWidget::initPBO()
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
void DensitySOMWidget::initSplitPBO()
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
void DensitySOMWidget::initializeGL()
{
	initPBO();
	initLogPBO();
	initSplitPBO();

	cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
	cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );

}
void DensitySOMWidget::resizeGL( int x, int y )
{

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 2.0, 0.0, 2.0, 0.0, 1.0);


}
void DensitySOMWidget::paintGL()
{
	if (som.RUN_CYCLE && som.counter < som.host_T){
		som.counter++;
		som.updateWeights();
		som.runCuda();
		som.updateConvergence();
	}

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// download image from PBO to OpenGL texture
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture  (GL_TEXTURE_TYPE, displayRegTex);
	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_TYPE,0, 0, 0, width/2, height/2, GL_RGBA, GL_UNSIGNED_BYTE, 0);
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

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, split_pbo);
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

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, log_pbo);
	glBindTexture  (GL_TEXTURE_TYPE, display_log_tex);
	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_TYPE,
					0, 0, 0, width/2, height/2, GL_LUMINANCE, GL_UNSIGNED_INT, 0);
	glEnable(GL_TEXTURE_TYPE);

	// draw textured quad
	glBegin(GL_QUADS);
	glTexCoord2f(0    , height/2);  glVertex2f(0, 1);
	glTexCoord2f(width/2, height/2);  glVertex2f(1, 1);
	glTexCoord2f(width/2, 0     );  glVertex2f(1, 2);
	glTexCoord2f(0    , 0     );  glVertex2f(0, 2);
	glEnd();
	glDisable(GL_TEXTURE_TYPE);

}
void DensitySOMWidget::keyPressEvent( QKeyEvent *e )
{
	switch(e->key()){
    case Qt::Key_N:
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );

        som.updateWeights();
        som.runCuda();
        som.updateConvergence();
       	cutilSafeCall(cudaGLUnmapBufferObject(pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(split_pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(log_pbo) );

    	break;
    case Qt::Key_Escape:
    	close();
    	break;

	}

	QGLWidget::updateGL();
}
