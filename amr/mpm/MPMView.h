#pragma once

#include <QGraphicsView>
#include <QMouseEvent>
#include <QListWidget>

class MPMView : public QGraphicsView
{
public:
    MPMView();
    MPMView(QWidget* parent = 0);
protected:
    void resizeEvent(QResizeEvent *event);
    //void mousePressEvent(QMouseEvent *event);
    //void mouseReleaseEvent(QMouseEvent *event);
private:
    QFrame *popup;
};
