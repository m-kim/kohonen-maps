#ifndef QTSOMWIDGET_H
#define QTSOMWIDGET_H
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <QtOpenGL/QGLWidget>

class QTimer;

class QtSOMWidget:public QGLWidget
{
	Q_OBJECT
public:
	QtSOMWidget( int timerInterval=0, QWidget *parent=0, QGLWidget *shareWidget = 0);
	QSize minimumSizeHint() const;
	QSize sizeHint() const;
protected:
	//Got this from http://www.lighthouse3d.com/opengl/glsl/index.php?oglinfo
	// it prints out shader info (debugging!)
	void printProgramInfoLog(GLuint obj);

	//Got this from http://www.lighthouse3d.com/opengl/glsl/index.php?oglinfo
	// it prints out shader info (debugging!)
	void printShaderInfoLog(GLuint obj);

	virtual void initializeGL() = 0;
	virtual void resizeGL( int width, int height ) = 0;
	virtual void paintGL() = 0;
	virtual void keyPressEvent( QKeyEvent *e );
	virtual void timeOut();
protected slots:
	virtual void timeOutSlot();
private:
	QTimer *m_timer;
};

#endif
