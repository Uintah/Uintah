#include"Particle.h"

//default constructor, sets stuff to zero and resizes matrices
ParticleData::ParticleData()
{
	pos = disp = vel = b = {0.0, 0.0};
   	F = sigma = grad_v = {0.0, 0.0, 0.0, 0.0};
   	m = 0.0;
   	V = 0.0;
    y_mod = 0.0;
}

void ParticleData::Print() const
{
    cout << endl <<
            "Particle #" << id << endl <<
            "Inside element #" << e_id << endl <<
            "Position: [" << pos[x1] << ",\t" << pos[x2] << "]" << endl <<
            "Mass:" << m << endl <<
            "Velocity: [" << vel[x1] << ",\t" << vel[x2] << "]" << endl <<
            "Velocity grad: [" << grad_v[x1] << ",\t" << grad_v[x2] << "]" << endl <<
            "Deformation grad: [" << F[a11] << ",\t" << F[a12] << ";\t" << F[a21] << ",\t" << F[a22] << "]" << endl <<
            "Stress tensor: [" << sigma[a11] << ",\t" << sigma[a12] << ";\t" << sigma[a21] << ",\t" << sigma[a22] << "]" << endl;
}


QParticleItem::QParticleItem(qreal x, qreal y, QGraphicsItem* parent, qreal norm):QGraphicsEllipseItem(-norm,-norm,2.0*norm,2.0*norm,parent)
{
    setPos(x,y);
    size = norm;
    popup = new QFrame();
    popup->setWindowFlags(Qt::FramelessWindowHint);
    setAcceptHoverEvents(true);
    setBrush(QBrush(Qt::black));
    setPen(QPen(Qt::black));
    setTransformOriginPoint(mapFromScene(x,y));
}

void QParticleItem::SetData(const ParticleData *p)
{
    data = p;
    setScale(data->m);
}

void QParticleItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    int dx = 250, dy = 200, shift = 20;
    QPointF pnt = event->lastScreenPos();
    popup->setGeometry(pnt.x()+shift,pnt.y()-dy-shift,dx,dy);
    popup->show();
    QString s;
    QListWidget *list = new QListWidget(popup);
    list->addItem(s.sprintf("ID: %i", data->id));
    list->addItem(s.sprintf("pos: [%.2f, %.2f]", data->pos[x1], data->pos[x2]));
    list->addItem(s.sprintf("v: [%.2f, %.2f]", data->vel[x1], data->vel[x2]));
    list->addItem(s.sprintf("grad v: [%.2f, %.2f; %.2f, %.2f]", data->grad_v[a11], data->grad_v[a12], data->grad_v[a21], data->grad_v[a22]));
    list->addItem(s.sprintf("sigma: [%.2f, %.2f; %.2f, %.2f]", data->sigma[a11], data->sigma[a12], data->sigma[a21], data->sigma[a22]));
    list->addItem(s.sprintf("F: [%.2f, %.2f; %.2f, %.2f]", data->F[a11], data->F[a12], data->F[a21], data->F[a22]));
    list->addItem(s.sprintf("b: [%.2f, %.2f]", data->b[x1], data->b[x2]));
    list->addItem(s.sprintf("m: %.2f", data->m));
    list->addItem(s.sprintf("V: %.2f", data->V));

    list->resize(dx,dy);

    list->show();
}

void QParticleItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    popup->hide();
}

void QParticleItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    data->Print();
}
