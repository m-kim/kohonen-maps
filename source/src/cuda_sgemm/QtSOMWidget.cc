#include "QtSOMWidget.h"
#include <QKeyEvent>
#include <qapplication.h>
#include <qtimer.h>

QtSOMWidget::QtSOMWidget( int timerInterval, QWidget *parent, char *name ) : QGLWidget( parent )
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
