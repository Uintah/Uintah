
/*
 *  Geom.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom.h>
#include <GL/glx.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream.h>

GeomObj::GeomObj()
: matl(0)
{
}

GeomObj::~GeomObj()
{
#if 0
    if(matl)
	delete matl;
#endif
}

ObjGroup::ObjGroup()
: objs(0, 100)
{
}

ObjGroup::~ObjGroup()
{
}

void ObjGroup::add(GeomObj* obj)
{
    objs.add(obj);
}

int ObjGroup::size()
{
    return objs.size();
}

void ObjGroup::draw()
{
    for (int i=0; i<objs.size(); i++)
	objs[i]->draw();
}

Triangle::Triangle(const Point& p1, const Point& p2, const Point& p3)
: p1(p1), p2(p2), p3(p3)
{
}

Triangle::~Triangle()
{
}

void Triangle::draw() {
    glColor3f(0, 1, 0);
    glBegin(GL_TRIANGLES);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glVertex3d(p3.x(), p3.y(), p3.z());
    glEnd();
}

GeomPt::GeomPt(const Point& p)
: p1(p)
{
}

GeomPt::~GeomPt() {
}

void GeomPt::draw() {
    glColor3f(0, 0, 1);
    glBegin(GL_POINTS);
    glVertex3d(p1.x(), p1.y(), p1.z());
}

