
/*
 *  GeomOpenGL.cc: Rendering for OpenGL windows
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/GeomOpenGL.h>
#include <Geom/Cone.h>
#include <Geom/Cylinder.h>
#include <Geom/Disc.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/HeadLight.h>
#include <Geom/Light.h>
#include <Geom/Line.h>
#include <Geom/PointLight.h>
#include <Geom/Polyline.h>
#include <Geom/Sphere.h>
#include <Geom/Tetra.h>
#include <Geom/Tri.h>
#include <Geom/TriStrip.h>
#include <Geom/View.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

#include <GL/gl.h>
#include <GL/glu.h>

DrawInfoOpenGL::DrawInfoOpenGL()
: current_matl(0)
{
    qobj=gluNewQuadric();
}

DrawInfoOpenGL::~DrawInfoOpenGL()
{
    gluDeleteQuadric(qobj);
}

void DrawInfoOpenGL::set_drawtype(DrawType dt)
{
    drawtype=dt;
    switch(drawtype){
    case DrawInfoOpenGL::WireFrame:
	gluQuadricNormals(qobj, GLU_NONE);
	gluQuadricDrawStyle(qobj, GLU_LINE);
	break;
    case DrawInfoOpenGL::Flat:
	gluQuadricNormals(qobj, GLU_NONE);
	gluQuadricDrawStyle(qobj, GLU_FILL);
	break;
    case DrawInfoOpenGL::Gouraud:
	gluQuadricNormals(qobj, GLU_FLAT);
	gluQuadricDrawStyle(qobj, GLU_FILL);
	break;
    case DrawInfoOpenGL::Phong:
	gluQuadricNormals(qobj, GLU_SMOOTH);
	gluQuadricDrawStyle(qobj, GLU_FILL);
	break;
    }
    
}
void DrawInfoOpenGL::set_matl(Material* matl)
{
    float color[4];
    matl->ambient.get_color(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
    matl->diffuse.get_color(color);
    glColor4fv(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    matl->specular.get_color(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
    matl->emission.get_color(color);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, matl->shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, color);
}

void DrawInfoOpenGL::push_matl(Material* matl)
{
    stack.push(matl);
    if(current_matl != matl){
	current_matl=matl;
	set_matl(matl);
    }
}

void DrawInfoOpenGL::pop_matl()
{
    stack.pop();
    if(stack.size() > 0){
	Material* top=stack.top();
	if(current_matl != top){
	    current_matl=top;
	    set_matl(top);
	}
    } else {
	current_matl=0;
    }
}

void GeomObj::draw(DrawInfoOpenGL* di)
{
    if(matl.get_rep())
	di->push_matl(matl.get_rep());
    if(lit && di->lighting && !di->currently_lit){
	di->currently_lit=1;
	glEnable(GL_LIGHTING);
    }
    if((!lit || !di->lighting) && di->currently_lit){
	di->currently_lit=0;
	glDisable(GL_LIGHTING);
    }
    if(di->pickmode && pick)
	glPushName((GLuint)pick);
    objdraw(di);
    if(di->pickmode && pick)
	glPopName();
    if(matl.get_rep())
	di->pop_matl();
}

void GeomCone::objdraw(DrawInfoOpenGL* di)
{
    glPushMatrix();
    glRotated(zrotangle, zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    di->polycount+=nu*nv;
    gluCylinder(di->qobj, bot_rad, top_rad, height, nu, nv);
    glPopMatrix();
}

void GeomCylinder::objdraw(DrawInfoOpenGL* di)
{
    glPushMatrix();
    glRotated(zrotangle, zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    di->polycount+=nu*nv;
    gluCylinder(di->qobj, rad, rad, height, nu, nv);
    glPopMatrix();
}

void GeomDisc::objdraw(DrawInfoOpenGL* di)
{
    glPushMatrix();
    glTranslated(cen.x(), cen.y(), cen.z());
    glRotated(zrotangle, zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=nu*nv;
    gluDisk(di->qobj, 0, rad, nu, nv);
    glPopMatrix();
}

void GeomGroup::objdraw(DrawInfoOpenGL* di)
{
    for (int i=0; i<objs.size(); i++)
	objs[i]->draw(di);
}

void GeomLine::objdraw(DrawInfoOpenGL* di) {
    di->polycount++;
    glBegin(GL_LINE_STRIP);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glEnd();
}

void GeomPolyline::objdraw(DrawInfoOpenGL* di) {
    di->polycount+=pts.size()-1;
    glBegin(GL_LINE_STRIP);
    for(int i=0;i<pts.size();i++){
	Point p(pts[i]);
	glVertex3d(p.x(), p.y(), p.z());
    }
    glEnd();
}

void GeomSphere::objdraw(DrawInfoOpenGL* di)
{
    glPushMatrix();
    glTranslated(cen.x(), cen.y(), cen.z());
    di->polycount+=nu*nv;
    gluSphere(di->qobj, rad, nu, nv);
    glPopMatrix();
}

void GeomTetra::objdraw(DrawInfoOpenGL* di) {
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
}

void GeomTri::objdraw(DrawInfoOpenGL* di) {
    di->polycount++;
    switch(di->get_drawtype()){
    case DrawInfoOpenGL::WireFrame:
	glBegin(GL_LINE_LOOP);
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	break;
    case DrawInfoOpenGL::Flat:
	glBegin(GL_TRIANGLES);
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	break;
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	glBegin(GL_TRIANGLES);
	glNormal3d(-n.x(), -n.y(), -n.z());
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	break;
    }
}

void GeomTriStrip::objdraw(DrawInfoOpenGL* di) {
    if(pts.size() <= 2)
	return;
    di->polycount+=pts.size()-2;
    switch(di->get_drawtype()){
    case DrawInfoOpenGL::WireFrame:
	{
	    glBegin(GL_LINES);
	    Point p1(pts[0]);
	    Point p2(pts[1]);
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    for(int i=2;i<pts.size();i+=2){
		Point p3(pts[i]);
		Point p4(pts[i+1]);
		glVertex3d(p3.x(), p3.y(), p3.z());
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p3.x(), p3.y(), p3.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
		glVertex3d(p4.x(), p4.y(), p4.z());
		glVertex3d(p3.x(), p3.y(), p3.z());
		glVertex3d(p4.x(), p4.y(), p4.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
		p1=p3;
		p2=p4;
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Flat:
	{
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int i=0;i<pts.size();i++){
		Point p(pts[i]);
		glVertex3d(p.x(), p.y(), p.z());
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	{
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int i=0;i<pts.size();i++){
		Vector n(norms[i]);
		glNormal3d(n.x(), n.y(), n.z());
		Point p(pts[i]);
		glVertex3d(p.x(), p.y(), p.z());
	    }
	    glEnd();
	}
	break;
    }
}

void PointLight::opengl_setup(const View&, DrawInfoOpenGL*, int& idx)
{
    int i=idx++;
    float f[4];
    f[0]=p.x(); f[1]=p.y(); f[2]=p.z(); f[3]=1.0;
    glLightfv(GL_LIGHT0+i, GL_POSITION, f);
    c.get_color(f);
    glLightfv(GL_LIGHT0+i, GL_DIFFUSE, f);
    glLightfv(GL_LIGHT0+i, GL_SPECULAR, f);
}

void HeadLight::opengl_setup(const View& view, DrawInfoOpenGL*, int& idx)
{
    int i=idx++;
    Point p(view.eyep);
    float f[4];
    f[0]=p.x(); f[1]=p.y(); f[2]=p.z(); f[3]=1.0;
    glLightfv(GL_LIGHT0+i, GL_POSITION, f);
    c.get_color(f);
    glLightfv(GL_LIGHT0+i, GL_DIFFUSE, f);
    glLightfv(GL_LIGHT0+i, GL_SPECULAR, f);
}

