#include"Solver.h"

///////////////////////////////////////////////////////////////////////////////
// NODE CLASS
///////////////////////////////////////////////////////////////////////////////
void NodeData::Print() const
{
    cout << endl <<
            "Node #" << id << endl <<
            "Position: [" << pos[x1] << ",\t" << pos[x2] << "]" << endl <<
            "Mass: " << mass << endl <<
            "Momentum: [" << momentum[x1] << ",\t" << momentum[x2] << "]" << endl <<
            "Velocity: [" << vel[x1] << ",\t" << vel[x2] << "]" << endl <<
            "Acceleration: [" << accel[x1] << ",\t" << accel[x2] << "]" << endl <<
            "Interior force: [" << f_int[x1] << ",\t" << f_int[x2] << "]" << endl <<
            "Exterior force: [" << f_ext[x1] << ",\t" << f_ext[x2] << "]";
}

Node::Node():hasImage(false),isActive(false),isRegular(true)
{
	data = new NodeData();
}

Node::Node(const Vec2D& v, const int i):hasImage(false),isActive(false),isRegular(true)
{
	data = new NodeData();
    data->pos = v;
    data->id = i;
}

Node::~Node()
{
	if(data) delete data;
    if(hasImage) delete image;
}

void Node::Reset()
{
	data->mass = 0.0;
	data->vel = {0.0, 0.0};
	data->accel = {0.0, 0.0};
	data->momentum = {0.0, 0.0};
	data->f_int = {0.0, 0.0};
	data->f_ext = {0.0, 0.0};
	isActive = false;
}

void Node::Interpolate()
{
    //if the node itself is active (affects any particles)
    if(isActive)
    {
        //nodes from which we interpolate the values
        Node* n1 = interp[0];
        Node* n2 = interp[1];
        //interpolating only from two active, regular nodes and only if the element containing hanging node
        //has any particles in it
        if(interp_el->GetNParticles() != 0 && (n1->isActive && n2->isActive && n1->isRegular && n2->isRegular))
        {
            /*
            cout << "\nNode " << data->id << " is interpolated from element " << interp_el->ID()
                 << " that has " << interp_el->GetNParticles() << " particles "
                 <<" using nodes #" << n1->data->id << " and #" << n2->data->id << endl;
                 */
            NodeData* d1 = n1->data;
            NodeData* d2 = n2->data;
            //computing distances between node2 and node1
            //and between current node and node1
            double l12 = Vec2DLength(d2->pos - d1->pos);
            double l1 = Vec2DLength(data->pos - d1->pos);
            assert(l12 != 0.0);
            //linear coefficient
            double a = l1/l12;
            //interpolating values
            //assuming that we'll do interpolation after the timestep for active nodes
            //we only need to interpolate acceleration and velocity
            data->accel = d1->accel + a * (d2->accel - d1->accel);
            data->vel = d1->vel + a * (d2->vel - d1->vel);
        }
    }
}

QNodeItem::QNodeItem(qreal x, qreal y, QGraphicsItem* parent, qreal norm):QGraphicsRectItem(x-norm,y-norm,2.0*norm,2.0*norm,parent)
{
    popup = new QFrame();
    popup->setWindowFlags(Qt::FramelessWindowHint);
    setAcceptHoverEvents(true);
    //setBrush(QBrush(Qt::black));
    setBrush(QBrush(Qt::white));
    QPen rect_pen(Qt::black);
    rect_pen.setWidth(1.5);
    setPen(rect_pen);
    setTransformOriginPoint(mapFromScene(x,y));
}

void QNodeItem::SetData(const NodeData *d)
{
    data = d;
}

void QNodeItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    int dx = 200, dy = 140, shift = 20;
    QPointF pnt = event->lastScreenPos();
    popup->setGeometry(pnt.x()+shift,pnt.y()-dy-shift,dx,dy);
    popup->show();
    QString s;
    QListWidget *list = new QListWidget(popup);

    list->addItem(s.sprintf("ID: %i", data->id));
    list->addItem(s.sprintf("pos: [%.1f, %.1f]", data->pos[x1], data->pos[x2]));
    list->addItem(s.sprintf("v: [%.1f, %.1f]", data->vel[x1], data->vel[x2]));
    list->addItem(s.sprintf("p: [%.1f, %.1f]", data->momentum[x1], data->momentum[x2]));
    list->addItem(s.sprintf("acc: [%.1f, %.1f]", data->accel[x1], data->accel[x2]));
    list->addItem(s.sprintf("f_int: [%.1f, %.1f]", data->f_int[x1], data->f_int[x2]));
    list->addItem(s.sprintf("f_ext: [%.1f, %.1f]", data->f_ext[x1], data->f_ext[x2]));
    list->addItem(s.sprintf("m: %.1f", data->mass));

    list->resize(dx,dy);

    list->show();
}

void QNodeItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    data->Print();
}

void QNodeItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    popup->hide();
}
