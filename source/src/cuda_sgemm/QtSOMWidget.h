#ifndef QTSOMWIDGET_H
#define QTSOMWIDGET_H
#include <QtOpenGL/qgl.h>

class QTimer;

class QtSOMWidget:public QGLWidget
{
	Q_OBJECT
public:
	QtSOMWidget( int timerInterval=0, QWidget *parent=0, char *name=0 );

protected:
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
