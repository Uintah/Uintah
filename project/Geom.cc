
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
#include <Geometry/BBox.h>

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
    bb.extend(obj->bbox());
}

int ObjGroup::size()
{
    return objs.size();
}

void ObjGroup::draw(DrawInfo* di)
{
    if(matl)
	di->push_matl(matl);
    for (int i=0; i<objs.size(); i++)
	objs[i]->draw(di);
    if(matl)
	di->pop_matl();
}

BBox ObjGroup::bbox()
{
    return bb;
}

Triangle::Triangle(const Point& p1, const Point& p2, const Point& p3)
: p1(p1), p2(p2), p3(p3), n(Cross(p3-p1, p2-p1))
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
}

Triangle::~Triangle()
{
}

void Triangle::draw(DrawInfo* di) {
    if(matl)
	di->push_matl(matl);
    switch(di->drawtype){
    case DrawInfo::WireFrame:
	glBegin(GL_LINE_LOOP);
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	break;
    case DrawInfo::Flat:
	glBegin(GL_TRIANGLES);
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	break;
    case DrawInfo::Gouraud:
    case DrawInfo::Phong:
	glBegin(GL_TRIANGLES);
	glNormal3d(n.x(), n.y(), n.z());
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	break;
    }
    if(matl)
	di->pop_matl();
}

BBox Triangle::bbox() {
    return bb;
}

Tetra::Tetra(const Point& p1, const Point& p2, const Point& p3, const Point& p4)
: p1(p1), p2(p2), p3(p3), p4(p4)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
    bb.extend(p4);
}

Tetra::~Tetra()
{
}

void Tetra::draw(DrawInfo* di) {
    if(matl)
	di->push_matl(matl);
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
    if(matl)
	di->pop_matl();
}

BBox Tetra::bbox() {
    return bb;
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv)
: cen(cen), rad(rad), nu(nu), nv(nv)
{
    bb.extend(cen, rad);
}

GeomSphere::~GeomSphere()
{
}

void GeomSphere::draw(DrawInfo* di)
{
    if(matl)
	di->push_matl(matl);
    SinCosTable u(nu, 0, 2.*Pi);
    SinCosTable v(nv, 0, Pi, rad);
    double cx=cen.x();
    double cy=cen.y();
    double cz=cen.z();
    int i, j;
    cerr << "drawtype=" << di->drawtype << endl;
    switch(di->drawtype){
    case DrawInfo::WireFrame:
	for(i=0;i<nu-1;i++){
	    glBegin(GL_LINE_STRIP);
		double x0=u.sin(i);
	    double y0=u.cos(i);
	    for(int j=0;j<nv;j++){
		double r0=v.sin(j);
		double z0=v.cos(j);
		glVertex3d(x0*r0+cx, y0*r0+cy, z0+cz);
	    }
	    glEnd();
	}
	for(j=1;j<nv-1;j++){
	    glBegin(GL_LINE_LOOP);
	    double r0=v.sin(j);
	    double z0=v.cos(j);
	    for(int i=0;i<nu-1;i++){
		double x0=u.sin(i);
		double y0=u.cos(i);
		glVertex3d(x0*r0+cx, y0*r0+cy, z0+cz);
	    }
	    glEnd();
	}
	break;
    case DrawInfo::Flat:
	for(i=0;i<nu-1;i++){
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
		glVertex3d(x0*r0+cx, y0*r0+cy, z0+cz);
		glVertex3d(x1*r0+cx, y1*r0+cy, z0+cz);
		glVertex3d(x0*r1+cx, y0*r1+cy, z1+cz);
		glVertex3d(x1*r1+cx, y1*r1+cy, z1+cz);
	    }
	    glEnd();
	}
	break;
    case DrawInfo::Gouraud:
	for(i=0;i<nu-1;i++){
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
		glNormal3d(-x0*r0, -y0*r0, -z0);
		glVertex3d(x0*r0+cx, y0*r0+cy, z0+cz);
		glVertex3d(x1*r0+cx, y1*r0+cy, z0+cz);
		glVertex3d(x0*r1+cx, y0*r1+cy, z1+cz);
		glVertex3d(x1*r1+cx, y1*r1+cy, z1+cz);
	    }
	    glEnd();
	}
	break;
    case DrawInfo::Phong:
	for(i=0;i<nu-1;i++){
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
		glNormal3d(-x0*r0, -y0*r0, -z0);
		glVertex3d(x0*r0+cx, y0*r0+cy, z0+cz);
		glNormal3d(-x1*r0, -y1*r0, -z0);
		glVertex3d(x1*r0+cx, y1*r0+cy, z0+cz);
		glNormal3d(-x0*r1, -y0*r1, -z1);
		glVertex3d(x0*r1+cx, y0*r1+cy, z1+cz);
		glNormal3d(-x1*r1, -y1*r1, -z1);
		glVertex3d(x1*r1+cx, y1*r1+cy, z1+cz);
	    }
	    glEnd();
	}
	break;
    }
    if(matl)
	di->pop_matl();
}

BBox GeomSphere::bbox() {
    return bb;
}

GeomPt::GeomPt(const Point& p)
: p1(p)
{
    bb.extend(p);
}

GeomPt::~GeomPt() {
}

void GeomPt::draw(DrawInfo* di) {
    if(matl)
	di->push_matl(matl);
    glBegin(GL_POINTS);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glEnd();
    if(matl)
	di->pop_matl();
}

BBox GeomPt::bbox() {
    return bb;
}

MaterialProp::MaterialProp(const Color& ambient, const Color& diffuse,
			   const Color& specular, double shininess)
: ambient(ambient), diffuse(diffuse), specular(specular),
  shininess(shininess), emission(0,0,0)
{
}

void MaterialProp::set(DrawInfo* di)
{
    if(this==di->current_matl)
	return;
    float color[4];
    di->current_matl=this;
    switch(di->drawtype){
    case DrawInfo::WireFrame:
    case DrawInfo::Flat:
	diffuse.get_color(color);
	glColor4fv(color);
	break;
    case DrawInfo::Gouraud:
    case DrawInfo::Phong:
	ambient.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
	diffuse.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
	specular.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
	emission.get_color(color);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, color);
	break;
    }
}

DrawInfo::DrawInfo()
: current_matl(0)
{
}

void DrawInfo::push_matl(MaterialProp* matl)
{
    stack.push(matl);
    matl->set(this);
}

void DrawInfo::pop_matl()
{
    stack.pop();
    if(stack.size()>0){
	stack.top()->set(this);
    }
}
