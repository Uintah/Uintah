
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
#include <Math/Trig.h>
#include <Math/TrigTable.h>
#include <iostream.h>

GeomObj::GeomObj()
: matl(0)
{
}

GeomObj::~GeomObj()
{
    if(matl)
	delete matl;
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
    if(matl){
	for (int i=0; i<objs.size(); i++){
	    matl->set_matl();
	    objs[i]->draw();
	}
    } else {
	for (int i=0; i<objs.size(); i++)
	    objs[i]->draw();
    }
}

Triangle::Triangle(const Point& p1, const Point& p2, const Point& p3)
: p1(p1), p2(p2), p3(p3)
{
}

Triangle::~Triangle()
{
}

void Triangle::draw() {
    if(matl)matl->set_matl();
    glBegin(GL_TRIANGLES);
    Vector e1(p3-p1);
    Vector e2(p2-p1);
    Vector n(Cross(e1, e2));
    glNormal3d(n.x(), n.y(), n.z());
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glVertex3d(p3.x(), p3.y(), p3.z());
    glEnd();
}

Tetra::Tetra(const Point& p1, const Point& p2, const Point& p3, const Point& p4)
: p1(p1), p2(p2), p3(p3), p4(p4)
{
}

Tetra::~Tetra()
{
}

void Tetra::draw() {
    if(matl)matl->set_matl();
    glBegin(GL_LINE_STRIP);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glVertex3d(p3.x(), p3.y(), p3.z());
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p4.x(), p4.y(), p4.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glVertex3d(p3.x(), p3.y(), p3.z());
    glVertex3d(p4.x(), p4.y(), p4.z());
    glEnd();
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv)
: cen(cen), rad(rad), nu(nu), nv(nv)
{
}

GeomSphere::~GeomSphere()
{
}

void GeomSphere::draw()
{
    if(matl)matl->set_matl();
    SinCosTable u(nu, 0, 2.*Pi);
    SinCosTable v(nv, -Pi/2., Pi/2., rad);
    for(int i=0;i<nu-1;i++){
	glBegin(GL_TRIANGLE_STRIP);
	double x0=u.sin(i);
	double y0=u.cos(i);
	double x1=u.sin(i+1);
	double y1=u.cos(i+1);
	for(int j=0;j<nv-1;j++){
	    double r0=v.sin(j);
	    double z0=v.cos(j);
	    double r1=v.sin(j+1);
	    double z1=v.cos(j+1);
	    glVertex3d(x0*r0, y0*r0, z0);
	    glVertex3d(x1*r0, y1*r0, z0);
	    glVertex3d(x0*r1, y0*r1, z1);
	    glVertex3d(x1*r1, y1*r1, z1);
	}
	glEnd();
    }
}

GeomPt::GeomPt(const Point& p)
: p1(p)
{
}

GeomPt::~GeomPt() {
}

void GeomPt::draw() {
    if(matl)matl->set_matl();
    glBegin(GL_POINTS);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glEnd();
}

void MaterialProp::set()
{
    float color[3];
    ambient.get_color(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
    diffuse.get_color(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    specular.get_color(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
    emission.get_color(color);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, color);
}
