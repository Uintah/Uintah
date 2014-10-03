#pragma once

#include"Typedefs.h"
#include"Utils.h"
#include<iostream>
#include<QFrame>
#include<QListWidget>
#include<QGraphicsSceneHoverEvent>
#include<QGraphicsScene>
#include<QGraphicsRectItem>
#include<QGraphicsEllipseItem>
#include"Particle.h"
using namespace std;

class NodeData;

class QNodeItem: public QGraphicsRectItem
{
public:
    //norm is the size normalization parameter, default is set to 10 (radius of the particle)
    QNodeItem(qreal x, qreal y, QGraphicsItem* parent = 0, qreal norm = 5.0);
    void SetData(const NodeData *p);
protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
private:
    int id;
    const NodeData *data;
    //can insert popup window text here
    QFrame *popup;
};

struct NodeData
{
	//physical values
    void Print() const;
	double mass;
	Vec2D momentum;
	Vec2D vel;
	Vec2D accel;
	Vec2D f_int;
	Vec2D f_ext;
    //id and position
    unsigned int id;
    Vec2D pos;
};

struct Node
{
	Node();
	~Node();
	Node(const Vec2D& v, const int i);
	void Reset();//reset all the the values to zero before the next time step
	void Interpolate();
    //setting node image
    inline void SetImage(QNodeItem* i){image = i; image->SetData(data); hasImage = true;}
    inline QNodeItem* GetImage(){return image;}
	bool isRegular;//regular or hanging
	//at each step node values are reset and all nodes are set to be inactive (zero)
	//if a data projected to the node from a particle (any of the elements containing node
	//also contains particles) the the node becomes active, otherwise it stays inactive
	bool isActive;

    ElementPtrList neighbors;

	InterpolationData interp;

    //graphical representation of node
    QNodeItem *image;
    bool hasImage;
	//data values on the node
	NodeData *data;
};

