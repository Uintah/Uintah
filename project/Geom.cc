
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
#include <NotFinished.h>
#include <iostream.h>
#include <Geometry/BBox.h>

GeomObj::GeomObj()
: matl(0)
{
}

GeomObj::GeomObj(const GeomObj& copy)
: matl(copy.matl)
{
}

GeomObj::~GeomObj()
{
    if(matl)
	delete matl;
}

void GeomObj::set_pick(GeomPick* _pick)
{
    pick=_pick;
}

void GeomObj::set_matl(MaterialProp* _matl)
{
    matl=_matl;
}

ObjGroup::ObjGroup()
: objs(0, 100)
{
}

ObjGroup::ObjGroup(const ObjGroup& copy)
: GeomObj(copy), bb(copy.bb)
{
    objs.grow(copy.objs.size());
    for(int i=0;i<objs.size();i++)
	objs[i]=copy.objs[i]->clone();
}

ObjGroup::~ObjGroup()
{
    for(int i=0;i<objs.size();i++)
	delete objs[i];
}

void ObjGroup::add(GeomObj* obj)
{
    objs.add(obj);
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

GeomObj* ObjGroup::clone()
{
    return new ObjGroup(*this);
}

void ObjGroup::get_bounds(BBox& in_bb)
{
    if(!bb.valid()){
	for(int i=0;i<objs.size();i++)
	    objs[i]->get_bounds(bb);
    }
    if(bb.valid())
	in_bb.extend(bb);
}

Triangle::Triangle(const Point& p1, const Point& p2, const Point& p3)
: p1(p1), p2(p2), p3(p3), n(Cross(p3-p1, p2-p1))
{
}

Triangle::Triangle(const Triangle &copy)
: p1(copy.p1), p2(copy.p2), p3(copy.p3), n(copy.n)
{
}

Triangle::~Triangle()
{
}

void Triangle::draw(DrawInfo* di) {
    if(matl)
	di->push_matl(matl);
    di->polycount++;
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

GeomObj* Triangle::clone()
{
    return new Triangle(*this);
}

void Triangle::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
}

Tetra::Tetra(const Point& p1, const Point& p2, const Point& p3, const Point& p4)
: p1(p1), p2(p2), p3(p3), p4(p4)
{
}

Tetra::Tetra(const Tetra& copy)
: p1(copy.p1), p2(copy.p2), p3(copy.p3), p4(copy.p4)
{
}

Tetra::~Tetra()
{
}

void Tetra::draw(DrawInfo* di) {
    if(matl)
	di->push_matl(matl);
    di->polycount+=4;
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

GeomObj* Tetra::clone()
{
    return new Tetra(*this);
}

void Tetra::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
    bb.extend(p4);
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv)
: cen(cen), rad(rad), nu(nu), nv(nv)
{
}

GeomSphere::GeomSphere(const GeomSphere& copy)
: cen(copy.cen), rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
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
    di->polycount+=nu*nv;
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

GeomObj* GeomSphere::clone()
{
    return new GeomSphere(*this);
}

void GeomSphere::get_bounds(BBox& bb)
{
    bb.extend(cen, rad);
}

GeomPt::GeomPt(const Point& p)
: p1(p)
{
}

GeomPt::GeomPt(const GeomPt& copy)
: p1(copy.p1)
{
}

GeomPt::~GeomPt() {
}

void GeomPt::draw(DrawInfo* di) {
    if(matl)
	di->push_matl(matl);
    di->polycount++;
    glBegin(GL_POINTS);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glEnd();
    if(matl)
	di->pop_matl();
}

GeomObj* GeomPt::clone()
{
    return new GeomPt(*this);
}

void GeomPt::get_bounds(BBox& bb)
{
    bb.extend(p1);
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

GeomPick::GeomPick(const Vector& v1)
: directions(2)
{
    directions[0]=v1;
    directions[1]=-v1;
}

GeomPick::~GeomPick()
{
}

void GeomPick::set_highlight(MaterialProp* matl)
{
    hightlight=matl;
}

GeomCylinder::GeomCylinder(const Point& bottom, const Point& top,
			   double rad, int nu, int nv)
: bottom(bottom), top(top), rad(rad), nu(nu), nv(nv), axis(top-bottom)
{
    if(axis.length2() < 1.e-6){
	cerr << "Degenerate cylinder!\n";
    }
    Vector v0(Cross(axis, Vector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(axis, Vector(0,1,0));
    }
    v1=Cross(axis, v0);
    v1.normalize();
    v1*=rad;
    v2=Cross(axis, v1);
    v2.normalize();
    v2*=rad;
}

GeomCylinder::GeomCylinder(const GeomCylinder& copy)
: bottom(copy.bottom), top(copy.top), rad(copy.rad), nu(copy.nu),
  nv(copy.nv), axis(copy.axis), v1(copy.v1), v2(copy.v2)
{
}

GeomCylinder::~GeomCylinder()
{
}

GeomObj* GeomCylinder::clone()
{
    return new GeomCylinder(*this);
}

void GeomCylinder::get_bounds(BBox& bb)
{
    NOT_FINISHED("GeomCylinder::get_bounds");
    bb.extend(bottom);
    bb.extend(top);
}

void GeomCylinder::draw(DrawInfo* di)
{
    if(matl)
	di->push_matl(matl);
    SinCosTable u(nu, 0, 2.*Pi);
    int i,j;
    di->polycount+=nu*nv;
    switch(di->drawtype){
    case DrawInfo::WireFrame:
	for(i=0;i<=nv;i++){
	    double z=double(i)/double(nv);
	    Vector up(axis*z);
	    Point bot_up(bottom+up);
	    glBegin(GL_LINE_LOOP);
	    for(int j=0;j<nu-1;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Point p(bot_up+v1*d1+v2*d2);
		glVertex3d(p.x(), p.y(), p.z());
	    }
	    glEnd();
	}
	glBegin(GL_LINES);
	for(j=0;j<nu-1;j++){
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    Point p1(bottom+v1*d1+v2*d2);
	    Point p2(p1+axis);
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	}
	glEnd();
	break;
    case DrawInfo::Gouraud:
    case DrawInfo::Phong:
    case DrawInfo::Flat:
	NOT_FINISHED("GeomCylinder::Draw");
	break;
    }
    if(matl)
	di->pop_matl();
}

GeomCone::GeomCone(const Point& bottom, const Point& top,
		   double bot_rad, double top_rad, int nu, int nv)
: bottom(bottom), top(top), bot_rad(bot_rad),
  top_rad(top_rad), nu(nu), nv(nv), axis(top-bottom)
{
    if(axis.length2() < 1.e-6){
	cerr << "Degenerate Cone!\n";
    } else {
	axis.normalize();
    }
    Vector v0(Cross(axis, Vector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(axis, Vector(0,1,0));
    }
    v1=Cross(axis, v0);
    v1.normalize();
    v2=Cross(axis, v1);
    v2.normalize();
}

GeomCone::GeomCone(const GeomCone& copy)
: bottom(copy.bottom), top(copy.top), top_rad(copy.top_rad),
  bot_rad(copy.bot_rad), nu(copy.nu), nv(copy.nv),
  v1(copy.v1), v2(copy.v2), axis(copy.axis)
{
}

GeomCone::~GeomCone()
{
}

GeomObj* GeomCone::clone()
{
    return new GeomCone(*this);
}

void GeomCone::get_bounds(BBox& bb)
{
    NOT_FINISHED("GeomCone::get_bounds");
    bb.extend(bottom);
    bb.extend(top);
}

void GeomCone::draw(DrawInfo* di)
{
    if(matl)
	di->push_matl(matl);
    switch(di->drawtype){
    case DrawInfo::WireFrame:
    case DrawInfo::Gouraud:
    case DrawInfo::Phong:
    case DrawInfo::Flat:
	NOT_FINISHED("GeomCone::draw");
	break;
    }
    if(matl)
	di->push_matl(matl);
}

GeomDisc::GeomDisc(const Point& cen, const Vector& normal,
		   double rad, int nu, int nv)
: cen(cen), normal(normal), rad(rad), nu(nu), nv(nv)
{
    if(normal.length2() < 1.e-6){
	cerr << "Degenerate normal on Disc!\n";
    }
    Vector v0(Cross(normal, Vector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(normal, Vector(0,1,0));
    }
    v1=Cross(normal, v0);
    v1.normalize();
    v2=Cross(normal, v1);
    v2.normalize();
}

GeomDisc::GeomDisc(const GeomDisc& copy)
: cen(copy.cen), normal(copy.normal), rad(copy.rad), nu(copy.nu),
  nv(copy.nv), v1(copy.v1), v2(copy.v2)
{
}

GeomDisc::~GeomDisc()
{
}

GeomObj* GeomDisc::clone()
{
    return new GeomDisc(*this);
}

void GeomDisc::get_bounds(BBox& bb)
{
    NOT_FINISHED("GeomDisc::get_bounds");
    bb.extend(cen);
}

void GeomDisc::draw(DrawInfo* di)
{
    if(matl)
	di->push_matl(matl);
    SinCosTable u(nu, 0, 2.*Pi);
    int i,j;
    di->polycount+=nu*nv;
    switch(di->drawtype){
    case DrawInfo::WireFrame:
	for(i=1;i<nv+1;i++){
	    glBegin(GL_LINE_LOOP);
	    double r=rad*double(i)/double(nv);
	    for(int j=0;j<nu-1;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Point p(cen+v1*(d1*r)+v2*(d2*r));
		glVertex3d(p.x(), p.y(), p.z());
	    }
	    glEnd();
	}
	glBegin(GL_LINES);
	for(j=0;j<nu-1;j++){
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    Point p(cen+v1*(d1*rad)+v2*(d2*rad));
	    glVertex3d(cen.x(), cen.y(), cen.z());
	    glVertex3d(p.x(), p.y(), p.z());
	}
	glEnd();
	break;
    case DrawInfo::Gouraud:
    case DrawInfo::Phong:
	glNormal3d(normal.x(), normal.y(), normal.z());
	// Trickle through...
    case DrawInfo::Flat:
	for(i=0;i<nv;i++){
	    glBegin(GL_TRIANGLE_STRIP);
	    double r1=rad*double(i)/double(nv);
	    double r2=rad*double(i+1)/double(nv);
	    for(int j=0;j<nu;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Point p1(cen+v1*(d1*r1)+v2*(d2*r1));
		Point p2(cen+v1*(d1*r2)+v2*(d2*r2));
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
	    glEnd();
	}
	break;
    }
    if(matl)
	di->pop_matl();
}
