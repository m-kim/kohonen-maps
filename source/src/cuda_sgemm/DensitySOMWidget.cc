#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include "DensitySOMWidget.h"
#include "Matrix.h"
#include "SOM.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <QtGui/QKeyEvent>
#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB

unsigned int DensitySOMWidget::shared_list_object = 0;

DensitySOMWidget::DensitySOMWidget( int timerInterval, QWidget *parent,QGLWidget *shareWidget):QtSOMWidget(0, parent, shareWidget)
{
	pbo = 0;
	log_pbo = 0;
	split_pbo = 0;
	displayRegTex = 0;
	displaySplitTex = 0;
	display_log_tex = 0;
	hist_vbo = 0;

	width = 1024;
	height = 1024;

	vert_shader = 0;
	frag_shader = 0;
	geo_shader = 0;
	prog = 0;
	geo_shader_prog = readShader("hist.geo");
	frag_shader_prog = readShader("hist.frag");
	vert_shader_prog = readShader("hist.vert");
	scale = 1;
}

const char* DensitySOMWidget::readShader(std::string name)
{
	std::ifstream file;
	std::stringstream filename;
	filename << SRC_PATH << "/src/cuda_sgemm/" << name;

	file.open(filename.str().c_str(), std::ifstream::in);
	std::string str;
	std::stringstream complete_file;
	int line_count = 0;
	int row = 0;
	if (!file.good()){
		std::cout << "shader file missing!  " << filename.str() << std::endl;
		exit(-1);
	}
	while (file.good()){
		str.clear();
		getline(file, str);
		complete_file << str << "\n";
	}

	char *tmp = new char[complete_file.str().length() + 1];
	strcpy(tmp, complete_file.str().c_str());
	file.close();
	return tmp;
}
void DensitySOMWidget::setupCuda(ORDERED_MATRIX<MATRIX_TYPE, HOST, COLUMN_MAJOR> &ww,
		ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> &data,
		unsigned int *labels)
{
	som.setupCuda(ww, data, labels,(uint *)d_regular_output, d_split_output,d_log_output, d_hist_output);
}

void DensitySOMWidget::initVBO()
{
    if (hist_vbo) {
        // delete old buffer
        cutilSafeCall(cudaGLUnregisterBufferObject(hist_vbo));
        glDeleteBuffersARB(1, &hist_vbo);
    }


    // create pixel buffer object for display
    glGenBuffersARB(1, &hist_vbo);
    glBindBufferARB(GL_ARRAY_BUFFER, hist_vbo);
    glBufferDataARB(GL_ARRAY_BUFFER, IMAGE_XxY*sizeof( unsigned int ) * 3, 0, GL_STREAM_COPY);

	glBindBufferARB(GL_ARRAY_BUFFER, 0);

    cutilSafeCall(cudaGLRegisterBufferObject(hist_vbo));

//    if (display_log_tex) {
//        glDeleteTextures(1, &display_log_tex);
//    }
//    glGenTextures(1, &display_log_tex);
//    glBindTexture  (GL_TEXTURE_TYPE, display_log_tex);
//    glTexImage2D   (GL_TEXTURE_TYPE, 0, GL_LUMINANCE, width/2, height/2, 0, GL_LUMINANCE, GL_UNSIGNED_INT, NULL);
//    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//    glBindTexture  (GL_TEXTURE_TYPE, 0);

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
void DensitySOMWidget::createShader()
{
	vert_shader = glCreateShader(GL_VERTEX_SHADER);
	frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	geo_shader = glCreateShader(GL_GEOMETRY_SHADER_EXT);

	glShaderSource(vert_shader, 1, &vert_shader_prog, NULL);
	glShaderSource(frag_shader, 1, &frag_shader_prog, NULL);
	glShaderSource(geo_shader, 1, &geo_shader_prog, NULL);

	glCompileShader(vert_shader);
	glCompileShader(frag_shader);
	glCompileShader(geo_shader);

	prog = glCreateProgram();

	glAttachShader(prog, vert_shader);
	glAttachShader(prog, frag_shader);
	glAttachShader(prog, geo_shader);

	glProgramParameteriEXT(prog,GL_GEOMETRY_INPUT_TYPE_EXT,GL_POINTS);
	glProgramParameteriEXT(prog,GL_GEOMETRY_OUTPUT_TYPE_EXT,GL_TRIANGLE_STRIP);
	int temp;
	glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT,&temp);
	glProgramParameteriEXT(prog,GL_GEOMETRY_VERTICES_OUT_EXT,temp);

	glLinkProgram(prog);
	glUseProgram(prog);

	if (som.DEBUG_PRINT){
		printShaderInfoLog(vert_shader);
		printShaderInfoLog(frag_shader);
		printShaderInfoLog(geo_shader);

		printProgramInfoLog(prog);
	}
}
void DensitySOMWidget::initializeGL()
{
	glEnable (GL_DEPTH_TEST);
//	glEnable (GL_LIGHTING);
//	glEnable (GL_LIGHT0);
	initPBO();
	initLogPBO();
	initSplitPBO();
	initVBO();

	cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
	cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );
	cutilSafeCall( cudaGLMapBufferObject((void**)&d_hist_output, hist_vbo) );

	if (shared_list_object == 0){
		shared_list_object = createListObject();
	}
	//createShader();
}

void DensitySOMWidget::resizeGL( int x, int y )
{

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1.0);
}

void DensitySOMWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//gluLookAt(0,0,1, 0,0,0, 0,1,0);
	if (som.RUN_CYCLE && som.counter < som.host_T){
		som.counter++;
		som.updateWeights();
		som.runCuda();
		som.updateConvergence();
	}

	glCallList(shared_list_object);

	std::cout << gluErrorString(glGetError()) << std::endl;
	// download image from PBO to OpenGL texture


//	glColor3f(1,0,0);
//
//	float light[] = {0,0,1};
//	glLightfv(GL_LIGHT0, GL_POSITION, light);
//	glBindBufferARB(GL_ARRAY_BUFFER, hist_vbo);         // for vertex coordinates
//	glEnableClientState(GL_VERTEX_ARRAY);                 // activate vertex coords array
//	glVertexPointer(3, GL_INT, 0, 0);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//	glPushMatrix();
//	glTranslatef(64,64, -128);
//    glRotatef(rot_x, 1.0f, 0.0f, 0.0f);
//    glRotatef(rot_y, 0.0f, 1.0f, 0.0f);
//    glRotatef(rot_z, 0.0f, 0.0f, 1.0f);
//
//    glScalef(scale * 1,scale * 1, scale * .5);
//	glDrawArrays(GL_POINTS, 0, IMAGE_XxY * 3);
//	glPopMatrix();
//	glDisableClientState(GL_VERTEX_ARRAY);
//	glDisableClientState(GL_COLOR_ARRAY);
//
//	// bind with 0, so, switch back to normal pointer operation
//	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
}

void DensitySOMWidget::keyPressEvent( QKeyEvent *e )
{
	switch(e->key()){
    case Qt::Key_N:
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_regular_output, pbo) );
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_split_output, split_pbo) );
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_log_output, log_pbo) );
       	cutilSafeCall( cudaGLMapBufferObject((void**)&d_hist_output, hist_vbo) );

        som.updateWeights();
        som.runCuda();
        som.updateConvergence();
       	cutilSafeCall(cudaGLUnmapBufferObject(pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(split_pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(log_pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(hist_vbo) );

    	break;
    default:
		QtSOMWidget::keyPressEvent(e);
	}

	QGLWidget::updateGL();
}

void DensitySOMWidget::mousePressEvent(QMouseEvent *e)
{
    anchor = e->pos();
}

void DensitySOMWidget::mouseMoveEvent(QMouseEvent *e)
{
    QPoint diff = e->pos() - anchor;

    if (e->buttons() & Qt::LeftButton) {
        rot_x += diff.y()/5.0f;
        rot_y += diff.x()/5.0f;
    } else if (e->buttons() & Qt::RightButton) {
        rot_z += diff.x()/5.0f;
    } else if (e->buttons() & Qt::MidButton){
    	scale -= diff.y()/5.0f;
	}

    anchor = e->pos();
   	QGLWidget::updateGL();
}

void DensitySOMWidget::unMap()
{
	if (som.RUN_DISPLAY){
		cutilSafeCall( cudaGLUnmapBufferObject(split_pbo) );
		cutilSafeCall( cudaGLUnmapBufferObject(log_pbo) );
		cutilSafeCall( cudaGLUnmapBufferObject(pbo) );
       	cutilSafeCall(cudaGLUnmapBufferObject(hist_vbo) );
	}
}

GLuint DensitySOMWidget::createListObject()
{

	GLuint list = glGenLists(1);
	glNewList(list, GL_COMPILE);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture  (GL_TEXTURE_TYPE, displayRegTex);
//	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
//
	glTexSubImage2D(GL_TEXTURE_TYPE,0, 0, 0, width/2, height/2, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glEnable(GL_TEXTURE_TYPE);
	glDisable(GL_DEPTH_TEST);
	glBegin(GL_QUADS);
	glTexCoord2f(0    , height/2);  glVertex2f(0, 0);
	glTexCoord2f(width/2, height/2);  glVertex2f(1, 0);
	glTexCoord2f(width/2, 0     );  glVertex2f(1, 1);
	glTexCoord2f(0    , 0     );  glVertex2f(0, 1);
	glEnd();
	glDisable(GL_TEXTURE_TYPE);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

//	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, split_pbo);
//	glBindTexture  (GL_TEXTURE_TYPE, displaySplitTex);
//	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
//	glTexSubImage2D(GL_TEXTURE_TYPE,
//					0, 0, 0, width/2, height/2, GL_RGBA, GL_UNSIGNED_BYTE, 0);
//	glEnable(GL_TEXTURE_TYPE);
//	glBegin(GL_QUADS);
//	glTexCoord2f(0    , height/2);  glVertex2f(1, 0);
//	glTexCoord2f(width/2, height/2);  glVertex2f(2, 0);
//	glTexCoord2f(width/2, 0     );  glVertex2f(2, 1);
//	glTexCoord2f(0    , 0     );  glVertex2f(1, 1);
//	glEnd();
//	glDisable(GL_TEXTURE_TYPE);
//
//
//	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, log_pbo);
//	glBindTexture  (GL_TEXTURE_TYPE, display_log_tex);
//	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
//	glTexSubImage2D(GL_TEXTURE_TYPE,
//					0, 0, 0, width/2, height/2, GL_LUMINANCE, GL_UNSIGNED_INT, 0);
//	glEnable(GL_TEXTURE_TYPE);
//	glBegin(GL_QUADS);
//	glTexCoord2f(0    , height/2);  glVertex2f(0, 1);
//	glTexCoord2f(width/2, height/2);  glVertex2f(1, 1);
//	glTexCoord2f(width/2, 0     );  glVertex2f(1, 2);
//	glTexCoord2f(0    , 0     );  glVertex2f(0, 2);
//	glEnd();
//	glDisable(GL_TEXTURE_TYPE);

    glEndList();

    return list;
}
