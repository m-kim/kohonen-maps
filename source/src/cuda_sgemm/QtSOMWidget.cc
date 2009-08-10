#include "QtSOMWidget.h"
#include <QKeyEvent>
#include <qapplication.h>
#include <qtimer.h>

QtSOMWidget::QtSOMWidget( int timerInterval, QWidget *parent, QGLWidget *shareWidget ) : QGLWidget( parent, shareWidget )
{
	if( timerInterval == 0 )
			m_timer = 0;
		else
		{
			m_timer = new QTimer( this );
			connect( m_timer, SIGNAL(timeout()), this, SLOT(timeOutSlot()) );
			m_timer->start( timerInterval );
		}
}

void QtSOMWidget::keyPressEvent( QKeyEvent *e )
{
	switch(e->key()){
    case Qt::Key_Escape:
    	close();
    	break;
	}

}

void QtSOMWidget::timeOut()
{
}

void QtSOMWidget::timeOutSlot()
{
	timeOut();
}


void QtSOMWidget::printProgramInfoLog(GLuint obj)
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

void QtSOMWidget::printShaderInfoLog(GLuint obj)
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

QSize QtSOMWidget::minimumSizeHint() const
{
    return QSize(128, 128);
}

QSize QtSOMWidget::sizeHint() const
{
    return QSize(512, 512);
}
