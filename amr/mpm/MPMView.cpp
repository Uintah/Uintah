#include "MPMView.h"

MPMView::MPMView()
{
    popup = new QFrame();
    popup->setWindowFlags(Qt::FramelessWindowHint);
}

MPMView::MPMView(QWidget* parent):QGraphicsView(parent)
{
    popup = new QFrame();
    popup->setWindowFlags(Qt::FramelessWindowHint);
}

void MPMView::resizeEvent(QResizeEvent *event)
{
    fitInView(sceneRect(),Qt::KeepAspectRatio);
}

/*
void MPMView::mousePressEvent(QMouseEvent *event)
{
    popup->hide();
    int dx = 200, dy = 50, shift = 10;
    QPoint pnt = event->globalPos();
    popup->setGeometry(pnt.x()+shift,pnt.y()-dy-shift,dx,dy);
    popup->show();
    QString s;
    QListWidget *list = new QListWidget(popup);
    QPointF scene_pos = mapToScene(event->pos());
    list->addItem(s.sprintf("pos: [%.1f, %.1f]", scene_pos.x(), scene_pos.y()));
    list->resize(dx,dy);
    list->show();
}

void MPMView::mouseReleaseEvent(QMouseEvent *event)
{
    popup->hide();
}
*/
