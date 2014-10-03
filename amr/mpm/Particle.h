#pragma once

#include"Utils.h"
#include"BoundingBox.h"
#include<string>
#include<iostream>
#include<QFrame>
#include<QListWidget>
#include<QGraphicsSceneHoverEvent>
#include<QGraphicsScene>
#include<QGraphicsEllipseItem>
using namespace std;
class QParticleItem;

struct ParticleData
{
	ParticleData();//default constructor that sets all entries to zero
    void Print() const;//printing particle data
	//particle information
    unsigned int id;//particle id
    unsigned int e_id;//id of element in which particle currently is
	Vec2D pos;   //position
	Vec2D disp;  //displacement
	Vec2D vel;   //velocity
	Vec2D b;    //acceleration due to body forces
	double m;     //mass
	double V;   //volume
	Mat2D grad_v; //velocity gradient
	Mat2D F;     //deformation gradient
	Mat2D sigma; //Cauchy stress tensor
    double y_mod; //Young's modulus
};

struct Particle
{
    Particle():image(nullptr), hasImage(false){};
    Particle(const ParticleData& d): data(d), image(nullptr), hasImage(false){};
    ParticleData data;
    QParticleItem* image;
    bool hasImage;
};

class QParticleItem: public QGraphicsEllipseItem
{
public:
    //QParticleItem(qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent = 0);
    //norm is the size normalization parameter, default is set to 5 (radius of the particle)
    QParticleItem(qreal x, qreal y, QGraphicsItem* parent = 0, qreal norm = 5.0);
    void SetData(const ParticleData *p);
    double R(){return (double)size;};
protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
private:
    int id;
    const ParticleData *data;
    //can insert popup window text here
    QFrame *popup;
    qreal size;
};


