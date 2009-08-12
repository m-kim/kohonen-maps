#ifndef HistSOMWidget_H
#define HistSOMWidget_H
#include "SOM.h"
#include "QtSOMWidget.h"
#include "Matrix.h"
#include <sstream>

class HistSOMWidget : public QtSOMWidget
{
public:
	HistSOMWidget( int timerInterval=0, QWidget *parent=0, QGLWidget *shareWidget = 0);
	~HistSOMWidget()
	{
	}
	SOM som;
	void setupCuda(ORDERED_MATRIX<MATRIX_TYPE, HOST, COLUMN_MAJOR> &ww,
			ORDERED_MATRIX<MATRIX_TYPE, HOST, ROW_MAJOR> &data,
			unsigned int *labels);

	void unMap();
	uint *d_split_output;
	uchar4 *d_regular_output;
	unsigned int *d_log_output;
	unsigned int *d_hist_output;
	void keyPressEvent( QKeyEvent *e );


protected:
	void initializeGL();
	void resizeGL( int width, int height );
	void paintGL();
    void mousePressEvent(QMouseEvent *);
    void mouseMoveEvent(QMouseEvent *);
	static GLuint sharedObject;
private:
	GLuint split_pbo, log_pbo;          // OpenGL pixel buffer object
	GLuint pbo;
	GLuint displayRegTex, displaySplitTex, display_log_tex;
	GLuint hist_vbo;
	int width, height;
    QPoint anchor;
    float rot_x, rot_y, rot_z, scale;

    GLuint vert_shader, frag_shader, geo_shader,prog;
	const char* geo_shader_prog, *frag_shader_prog, *vert_shader_prog;

    void initLogPBO();
	void initPBO();
	void initSplitPBO();
	void initVBO();
	void createShader();
	const char * readShader(std::string name);
	GLuint createObject();
};
#endif
