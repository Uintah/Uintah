
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
#include <Classlib/NotFinished.h>
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
#include <Geom/RenderMode.h>
#include <Geom/Sphere.h>
#include <Geom/Tetra.h>
#include <Geom/Tri.h>
#include <Geom/Tube.h>
#include <Geom/TriStrip.h>
#include <Geom/View.h>
#include <Geom/VCTri.h>
#include <Geom/VCTriStrip.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>
#include <Geometry/Plane.h>
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
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=nu*nv;
    gluCylinder(di->qobj, bot_rad, top_rad, height, nu, nv);
    glPopMatrix();
}

void GeomCylinder::objdraw(DrawInfoOpenGL* di)
{
    glPushMatrix();
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=nu*nv;
    gluCylinder(di->qobj, rad, rad, height, nu, nv);
    glPopMatrix();
}

void GeomDisc::objdraw(DrawInfoOpenGL* di)
{
    glPushMatrix();
    glTranslated(cen.x(), cen.y(), cen.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
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

// --------------------------------------------------

void GeomTube::objdraw(DrawInfoOpenGL* di) {
    Array1<Point> c1; 
    Array1<Point> c2; 
    Vector n1, n2; 
    Point  p1, p2; 
    for(int i=0; i<pts.size(); i++) {
       c1 = make_circle(pts[i], rad[i], normal[i]); 
       c2 = make_circle(pts[i+1], rad[i+1], normal[i+1]); 

       // draw triangle strips
            glBegin(GL_TRIANGLE_STRIP);
	    for(int j=0;j<c1.size();j++){

		n1 = c1[j]-pts[i];
		glNormal3d(n1.x(), n1.y(), n1.z());
		p1 = c1[j];
		glVertex3d(p1.x(), p1.y(), p1.z());

		n2 = c2[j]-pts[i+1];
		glNormal3d(n2.x(), n2.y(), n2.z());
		p2 = c2[j];
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
		n1 = c1[0]-pts[i];
		glNormal3d(n1.x(), n1.y(), n1.z());
		p1 = c1[0];
		glVertex3d(p1.x(), p1.y(), p1.z());

		n2 = c2[0]-pts[i+1];
		glNormal3d(n2.x(), n2.y(), n2.z());
		p2 = c2[0];
		glVertex3d(p2.x(), p2.y(), p2.z());

	    glEnd();
     }
  }

// --------------------------------------------------

void GeomRenderMode::objdraw(DrawInfoOpenGL* di)
{
    int save_lighting=di->lighting;
    NOT_FINISHED("GeomRenderMode");
    if(child){
	child->draw(di);
	// We don't put things back if no children...
	
    }
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
    switch(di->get_drawtype()){
    case DrawInfoOpenGL::WireFrame:
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
	break;
    case DrawInfoOpenGL::Flat:
	// this should be made into a tri-strip, but I couldn;t remember how...
	glBegin(GL_TRIANGLES);
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	glBegin(GL_TRIANGLES);
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p4.x(), p4.y(), p4.z());
	glEnd();
	glBegin(GL_TRIANGLES);
	glVertex3d(p4.x(), p4.y(), p4.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	glBegin(GL_TRIANGLES);
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p4.x(), p4.y(), p4.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	glEnd();
	break;
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	// this should be made into a tri-strip, but I couldn;t remember how...
	{
	    Plane PL1(p1,p2,p3); 
	    glBegin(GL_TRIANGLES);
	    glNormal3d(PL1.a, PL1.b, PL1.c);
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glEnd();
	    Plane PL2(p1,p2,p4); 
	    glBegin(GL_TRIANGLES);
	    glNormal3d(PL2.a, PL2.b, PL2.c);
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glEnd();
	    Plane PL3(p4,p2,p3); 
	    glBegin(GL_TRIANGLES);
	    glNormal3d(PL3.a, PL3.b, PL3.c);
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glEnd();
	    Plane PL4(p1,p4,p3); 
	    glBegin(GL_TRIANGLES);
	    glNormal3d(PL4.a, PL4.b, PL4.c);
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glEnd();
	    break;
	}
    }
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

void GeomVCTriStrip::objdraw(DrawInfoOpenGL* di) {
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
	{
	    glBegin(GL_TRIANGLE_STRIP);
	    di->push_matl(mmatl[0].get_rep());
	    for(int i=0;i<pts.size();i++){
		Vector n(norms[i]);
		glNormal3d(n.x(), n.y(), n.z());
		Point p(pts[i]);
		glVertex3d(p.x(), p.y(), p.z());
	    }
	    di->pop_matl();
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Phong:
	{
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int i=0;i<pts.size();i++){
		Vector n(norms[i]);
		glNormal3d(n.x(), n.y(), n.z());
		Point p(pts[i]);
		di->push_matl(mmatl[i].get_rep());
		glVertex3d(p.x(), p.y(), p.z());
		di->pop_matl();
	    }
	    glEnd();
	}
	break;
    }
}

void GeomVCTri::objdraw(DrawInfoOpenGL* di) {
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
	glBegin(GL_TRIANGLES);
	glNormal3d(-n.x(), -n.y(), -n.z());
	di->push_matl(m1.get_rep());
	glVertex3d(p1.x(), p1.y(), p1.z());
	glVertex3d(p2.x(), p2.y(), p2.z());
	glVertex3d(p3.x(), p3.y(), p3.z());
	di->pop_matl();
	glEnd();
	break;
    case DrawInfoOpenGL::Phong:
	glBegin(GL_TRIANGLES);
	glNormal3d(-n.x(), -n.y(), -n.z());
	di->push_matl(m1.get_rep());
	glVertex3d(p1.x(), p1.y(), p1.z());
	di->pop_matl();
	di->push_matl(m2.get_rep());
	glVertex3d(p2.x(), p2.y(), p2.z());
	di->pop_matl();
	di->push_matl(m3.get_rep());
	glVertex3d(p3.x(), p3.y(), p3.z());
	di->pop_matl();
	glEnd();
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

