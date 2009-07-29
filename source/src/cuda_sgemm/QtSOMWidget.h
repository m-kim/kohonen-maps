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
	QtSOMWidget( int timerInterval=0, QWidget *parent=0, char *name=0 );

protected:
	//Got this from http://www.lighthouse3d.com/opengl/glsl/index.php?oglinfo
		// it prints out shader info (debugging!)
		void printProgramInfoLog(GLuint obj)
		{
		    int infologLength = 0;
		    int charsWritten  = 0;
		    char *infoLog;
		    glGetProgramiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
		    if (infologLength > 0)
		    {
		        infoLog = (char *)malloc(infologLength);
		        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
			printf("printProgramInfoLog: %s\n",infoLog);
		        free(infoLog);
		    }else{
		 	printf("Program Info Log: OK\n");
		    }
		}

		//Got this from http://www.lighthouse3d.com/opengl/glsl/index.php?oglinfo
			// it prints out shader info (debugging!)
			void printShaderInfoLog(GLuint obj)
			{
			    int infologLength = 0;
			    int charsWritten  = 0;
			    char *infoLog;
			    glGetShaderiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
			    if (infologLength > 0)
			    {
			        infoLog = (char *)malloc(infologLength);
			        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
				printf("printShaderInfoLog: %s\n",infoLog);
			        free(infoLog);
			    }else{
				printf("Shader Info Log: OK\n");
			    }
			}

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
