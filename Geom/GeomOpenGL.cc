
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
#include <Geom/Material.h>
#include <Geom/Pick.h>
#include <Geom/PointLight.h>
#include <Geom/Polyline.h>
#include <Geom/Pt.h>
#include <Geom/RenderMode.h>
#include <Geom/Sphere.h>
#include <Geom/Switch.h>
#include <Geom/Tetra.h>
#include <Geom/Torus.h>
#include <Geom/Tri.h>
#include <Geom/Tube.h>
#include <Geom/TriStrip.h>
#include <Geom/View.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>
#include <Geometry/Plane.h>
#include <GL/gl.h>
#include <GL/glu.h>

#define MAX_MATL_STACK 100

void GeomObj::pre_draw(DrawInfoOpenGL* di, Material* matl, int lit)
{
    if(lit && di->lighting && !di->currently_lit){
	di->currently_lit=1;
	glEnable(GL_LIGHTING);
    }
    if((!lit || !di->lighting) && di->currently_lit){
	di->currently_lit=0;
	glDisable(GL_LIGHTING);
    }
    di->set_matl(matl);
}

DrawInfoOpenGL::DrawInfoOpenGL()
: current_matl(0)
{
    qobj=gluNewQuadric();
}

void DrawInfoOpenGL::reset()
{
    polycount=0;
    current_matl=0;
    ignore_matl=0;
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
    if(matl==current_matl || ignore_matl)
	return;
    current_matl=matl;
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

void GeomCappedCylinder::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    if(height < 1.e-6)return;
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=2*(nu-1)*(nv-1);
    gluCylinder(di->qobj, rad, rad, height, nu, nv);
    // Bottom endcap
    di->polycount+=2*(nu-1)*(nvdisc-1);
    gluDisk(di->qobj, 0, rad, nu, nvdisc);
    // Top endcap
    glTranslated(0, 0, height);
    di->polycount+=2*(nu-1)*(nvdisc-1);
    gluDisk(di->qobj, 0, rad, nu, nvdisc);
    glPopMatrix();
}

void GeomCone::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    if(height < 1.e-6)return;
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=2*(nu-1)*(nv-1);
    gluCylinder(di->qobj, bot_rad, top_rad, height, nu, nv);
    glPopMatrix();
}

void GeomCappedCone::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    if(height < 1.e-6)return;
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=2*(nu-1)*(nv-1);
    gluCylinder(di->qobj, bot_rad, top_rad, height, nu, nv);
    // Bottom endcap
    di->polycount+=2*(nu-1)*(nvdisc1-1);
    gluDisk(di->qobj, 0, bot_rad, nu, nvdisc1);
    // Top endcap
    glTranslated(0, 0, height);
    di->polycount+=2*(nu-1)*(nvdisc2-1);
    gluDisk(di->qobj, 0, top_rad, nu, nvdisc2);
    glPopMatrix();
}

void GeomContainer::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
    child->draw(di, matl, time);
}

void GeomCylinder::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    if(height < 1.e-6)return;
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=2*(nu-1)*(nv-1);
    gluCylinder(di->qobj, rad, rad, height, nu, nv);
    glPopMatrix();
}

void GeomDisc::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(cen.x(), cen.y(), cen.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=2*(nu-1)*(nv-1);
    gluDisk(di->qobj, 0, rad, nu, nv);
    glPopMatrix();
}

void GeomGroup::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
    for (int i=0; i<objs.size(); i++)
	objs[i]->draw(di, matl, time);
}

void GeomLine::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 0);
    di->polycount++;
    glBegin(GL_LINE_STRIP);
    glVertex3d(p1.x(), p1.y(), p1.z());
    glVertex3d(p2.x(), p2.y(), p2.z());
    glEnd();
}

void GeomMaterial::draw(DrawInfoOpenGL* di, Material* /* old_matl */, double time)
{
    child->draw(di, matl.get_rep(), time);
}

void GeomPick::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
    if(di->pickmode)
	glPushName((GLuint)this);
    if(selected){
	di->set_matl(highlight.get_rep());
	int old_ignore=di->ignore_matl;
	di->ignore_matl=1;
	child->draw(di, highlight.get_rep(), time);
	di->ignore_matl=old_ignore;
    } else {
	child->draw(di, matl, time);
    }
    if(di->pickmode)
	glPopName();
}

void GeomPolyline::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 0);
    di->polycount+=verts.size()-1;
    glBegin(GL_LINE_STRIP);
    for(int i=0;i<verts.size();i++){
	verts[i]->emit_point(di);
    }
    glEnd();
}

// --------------------------------------------------
void GeomPts::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 0);
    di->polycount+=pts.size();
    glBegin(GL_POINTS);
    for (int i=0; i<pts.size(); i++) {
	glVertex3d(pts[i].x(), pts[i].y(), pts[i].z());
    }
    glEnd();
}

void GeomTube::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    di->polycount+=(verts.size()-1)*2*20;
    Array1<Point> circle1;
    Array1<Point> circle2;
    Array1<Point>* p1=&circle1;
    Array1<Point>* p2=&circle2;
    SinCosTable tab(nu+1, 0, 2*Pi);
    make_circle(0, *p1, tab);
    if(di->get_drawtype() == DrawInfoOpenGL::WireFrame){
	glBegin(GL_LINE_LOOP);
	for(int j=0;j<p1->size();j++){
	    Point pt2((*p1)[j]);
	    glVertex3d(pt2.x(), pt2.y(), pt2.z());
	}
	glEnd();
    }
    for(int i=0; i<verts.size()-1; i++) {
	make_circle(i+1, *p2, tab);
	Array1<Point>& pp1=*p1;
	Array1<Point>& pp2=*p2;
	switch(di->get_drawtype()){
	case DrawInfoOpenGL::WireFrame:
	    {
		// Draw lines
		glBegin(GL_LINES);
		for(int j=0;j<nu;j++){
		    Point pt1(pp1[j]);
		    glVertex3d(pt1.x(), pt1.y(), pt1.z());
		    Point pt2(pp2[j]);
		    glVertex3d(pt2.x(), pt2.y(), pt2.z());
		}
		glEnd();
		glBegin(GL_LINE_LOOP);
		for(j=0;j<nu;j++){
		    Point pt2(pp2[j]);
		    glVertex3d(pt2.x(), pt2.y(), pt2.z());
		}
		glEnd();
	    }
	    break;
	case DrawInfoOpenGL::Flat:
	case DrawInfoOpenGL::Gouraud:
	case DrawInfoOpenGL::Phong:
	    {
		// draw triangle strips
		glBegin(GL_TRIANGLE_STRIP);
		Point cen1(verts[i]->p);
		Point cen2(verts[i+1]->p);
		for(int j=0;j<=nu;j++){
		    Point pt1(pp1[j]);
		    Vector n1(pt1-cen1);
		    verts[i]->emit_matl(di);
		    glNormal3d(n1.x(), n1.y(), n1.z());
		    glVertex3d(pt1.x(), pt1.y(), pt1.z());

		    Point pt2(pp2[j]);
		    Vector n2(pt2-cen2);
		    verts[i+1]->emit_matl(di);
		    glNormal3d(n2.x(), n2.y(), n2.z());
		    glVertex3d(pt2.x(), pt2.y(), pt2.z());
		}
		glEnd();
	    }
	}
	// Swith p1 and p2 pointers
	Array1<Point>* tmp=p1;
	p1=p2;
	p2=tmp;
    }
}

// --------------------------------------------------

void GeomRenderMode::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
//    int save_lighting=di->lighting;
    NOT_FINISHED("GeomRenderMode");
    if(child){
	child->draw(di, matl, time);
	// We don't put things back if no children...
	
    }
}    

void GeomSphere::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(cen.x(), cen.y(), cen.z());
    di->polycount+=2*(nu-1)*(nv-1);
    gluSphere(di->qobj, rad, nu, nv);
    glPopMatrix();
}

void GeomSwitch::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
   if(state)
      child->draw(di, matl, time);
}

void GeomTetra::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
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
	    Vector n1(Plane(p1, p2, p3).normal());
	    glBegin(GL_TRIANGLES);
	    glNormal3d(n1.x(), n1.y(), n1.z());
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glEnd();
	    Vector n2(Plane(p1, p2, p4).normal());
	    glBegin(GL_TRIANGLES);
	    glNormal3d(n2.x(), n2.y(), n2.z());
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glEnd();
	    Vector n3(Plane(p4, p2, p3).normal());
	    glBegin(GL_TRIANGLES);
	    glNormal3d(n3.x(), n3.y(), n3.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glEnd();
	    Vector n4(Plane(p1, p4, p3).normal());
	    glBegin(GL_TRIANGLES);
	    glNormal3d(n4.x(), n4.y(), n4.z());
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glEnd();
	    break;
	}
    }
}

void GeomTimeSwitch::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
    if(time >= tbeg && time < tend)
	child->draw(di, matl, time);
}

void GeomTorus::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(cen.x(), cen.y(), cen.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=2*(nu-1)*(nv-1);

    // Draw the torus
    SinCosTable tab1(nu, 0, 2*Pi);
    SinCosTable tab2(nv, 0, 2*Pi, rad2);
    SinCosTable tab2n(nv, 0, 2*Pi, rad2);
    int u,v;
    switch(di->get_drawtype()){

	
    case DrawInfoOpenGL::WireFrame:
	for(u=0;u<nu;u++){
	    double rx=tab1.sin(u);
	    double ry=tab1.cos(u);
	    glBegin(GL_LINE_LOOP);
	    for(v=1;v<nv;v++){
		double z=tab2.cos(v);
		double rad=rad1+tab2.sin(v);
		double x=rx*rad;
		double y=ry*rad;
		glVertex3d(x, y, z);
	    }
	    glEnd();
	}
	for(v=1;v<nv;v++){
	    double z=tab2.cos(v);
	    double rr=tab2.sin(v);
	    glBegin(GL_LINE_LOOP);
	    for(u=1;u<nu;u++){
		double rad=rad1+rr;
		double x=tab1.sin(u)*rad;
		double y=tab1.cos(u)*rad;
		glVertex3d(x, y, z);
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Flat:
	for(v=0;v<nv-1;v++){
	    double z1=tab2.cos(v);
	    double rr1=tab2.sin(v);
	    double z2=tab2.cos(v+1);
	    double rr2=tab2.sin(v+1);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(u=0;u<nu;u++){
		double r1=rad1+rr1;
		double r2=rad1+rr2;
		double xx=tab1.sin(u);
		double yy=tab1.cos(u);
		double x1=xx*r1;
		double y1=yy*r1;
		double x2=xx*r2;
		double y2=yy*r2;
		glVertex3d(x1, y1, z1);
		glVertex3d(x2, y2, z2);
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Gouraud:
	for(v=0;v<nv-1;v++){
	    double z1=tab2.cos(v);
	    double rr1=tab2.sin(v);
	    double z2=tab2.cos(v+1);
	    double rr2=tab2.sin(v+1);
	    double nr=-tab2n.sin(v);
	    double nz=-tab2n.cos(v);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(u=0;u<nu;u++){
		double r1=rad1+rr1;
		double r2=rad1+rr2;
		double xx=tab1.sin(u);
		double yy=tab1.cos(u);
		double x1=xx*r1;
		double y1=yy*r1;
		double x2=xx*r2;
		double y2=yy*r2;
		glNormal3d(nr*xx, nr*yy, nz);
		glVertex3d(x1, y1, z1);
		glVertex3d(x2, y2, z2);
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Phong:
	for(v=0;v<nv-1;v++){
	    double z1=tab2.cos(v);
	    double rr1=tab2.sin(v);
	    double z2=tab2.cos(v+1);
	    double rr2=tab2.sin(v+1);
	    double nr1=-tab2n.sin(v);
	    double nr2=-tab2n.sin(v+1);
	    double nz1=-tab2n.cos(v);
	    double nz2=-tab2n.cos(v+1);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(u=0;u<nu;u++){
		double r1=rad1+rr1;
		double r2=rad1+rr2;
		double xx=tab1.sin(u);
		double yy=tab1.cos(u);
		double x1=xx*r1;
		double y1=yy*r1;
		double x2=xx*r2;
		double y2=yy*r2;
		glNormal3d(nr1*xx, nr1*yy, nz1);
		glVertex3d(x1, y1, z1);
		glNormal3d(nr2*xx, nr2*yy, nz2);
		glVertex3d(x2, y2, z2);
	    }
	    glEnd();
	}
	break;	
    }
    glPopMatrix();
}

void GeomTri::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    di->polycount++;
    switch(di->get_drawtype()){
    case DrawInfoOpenGL::WireFrame:
	glBegin(GL_LINE_LOOP);
	verts[0]->emit_point(di);
	verts[1]->emit_point(di);
	verts[2]->emit_point(di);
	glEnd();
	break;
    case DrawInfoOpenGL::Flat:
	glBegin(GL_TRIANGLES);
	verts[0]->emit_point(di);
	verts[1]->emit_point(di);
	verts[2]->emit_point(di);
	glEnd();
	break;
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	glBegin(GL_TRIANGLES);
	glNormal3d(-n.x(), -n.y(), -n.z());
	verts[0]->emit_all(di);
	verts[1]->emit_all(di);
	verts[2]->emit_all(di);
	glEnd();
	break;
    }
}

void GeomTorusArc::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(cen.x(), cen.y(), cen.z());
    double matrix[16];
    matrix[0]=zero.x(); matrix[1]=zero.y(); matrix[2]=zero.z(); matrix[3]=0;
    matrix[4]=yaxis.x();matrix[5]=yaxis.y();matrix[6]=yaxis.z();matrix[7]=0;
    matrix[8]=axis.x(); matrix[9]=axis.y(); matrix[10]=axis.z();matrix[11]=0;
    matrix[12]=0;       matrix[13]=0;       matrix[14]=0;       matrix[15]=1;
    glMultMatrixd(matrix);
    di->polycount+=2*(nu-1)*(nv-1);

    // Draw the torus
    double a1=start_angle;
    double a2=start_angle-arc_angle;
    if(a1 > a2){
	double tmp=a1;
	a1=a2;
	a2=tmp;
    }
    SinCosTable tab1(nu, a1, a2);
    SinCosTable tab2(nv, 0, 2*Pi, rad2);
    SinCosTable tab2n(nv, 0, 2*Pi, rad2);
    int u,v;
    switch(di->get_drawtype()){

	
    case DrawInfoOpenGL::WireFrame:

	double srx=tab1.sin(0);
	double sry=tab1.cos(0);
	glBegin(GL_LINE_LOOP);
	for(v=1;v<nv;v++){
	    double sz=tab2.cos(v);
	    double srad=rad1+tab2.sin(v);
	    double sx=srx*srad;
	    double sy=sry*srad;
	    glVertex3d(sx, sy, sz);
	    glVertex3d(srx*rad1, sry*rad1, 0);
	}
	glEnd();

	srx=tab1.sin(nu-1);
	sry=tab1.cos(nu-1);
	glBegin(GL_LINE_LOOP);
	for(v=1;v<nv;v++){
	    double sz=tab2.cos(v);
	    double srad=rad1+tab2.sin(v);
	    double sx=srx*srad;
	    double sy=sry*srad;
	    glVertex3d(sx, sy, sz);
	    glVertex3d(srx*rad1, sry*rad1, 0);
	}
	glEnd();
	
	for(u=0;u<nu;u++){
	    double rx=tab1.sin(u);
	    double ry=tab1.cos(u);
	    glBegin(GL_LINE_LOOP);
	    for(v=1;v<nv;v++){
		double z=tab2.cos(v);
		double rad=rad1+tab2.sin(v);
		double x=rx*rad;
		double y=ry*rad;
		glVertex3d(x, y, z);
	    }
	    glEnd();
	}
	for(v=1;v<nv;v++){
	    double z=tab2.cos(v);
	    double rr=tab2.sin(v);
	    glBegin(GL_LINE_LOOP);
	    for(u=1;u<nu;u++){
		double rad=rad1+rr;
		double x=tab1.sin(u)*rad;
		double y=tab1.cos(u)*rad;
		glVertex3d(x, y, z);
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Flat:
	for(v=0;v<nv-1;v++){
	    double z1=tab2.cos(v);
	    double rr1=tab2.sin(v);
	    double z2=tab2.cos(v+1);
	    double rr2=tab2.sin(v+1);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(u=0;u<nu;u++){
		double r1=rad1+rr1;
		double r2=rad1+rr2;
		double xx=tab1.sin(u);
		double yy=tab1.cos(u);
		double x1=xx*r1;
		double y1=yy*r1;
		double x2=xx*r2;
		double y2=yy*r2;
		glVertex3d(x1, y1, z1);
		glVertex3d(x2, y2, z2);
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Gouraud:
	for(v=0;v<nv-1;v++){
	    double z1=tab2.cos(v);
	    double rr1=tab2.sin(v);
	    double z2=tab2.cos(v+1);
	    double rr2=tab2.sin(v+1);
	    double nr=-tab2n.sin(v);
	    double nz=-tab2n.cos(v);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(u=0;u<nu;u++){
		double r1=rad1+rr1;
		double r2=rad1+rr2;
		double xx=tab1.sin(u);
		double yy=tab1.cos(u);
		double x1=xx*r1;
		double y1=yy*r1;
		double x2=xx*r2;
		double y2=yy*r2;
		glNormal3d(nr*xx, nr*yy, nz);
		glVertex3d(x1, y1, z1);
		glVertex3d(x2, y2, z2);
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Phong:
	for(v=0;v<nv-1;v++){
	    double z1=tab2.cos(v);
	    double rr1=tab2.sin(v);
	    double z2=tab2.cos(v+1);
	    double rr2=tab2.sin(v+1);
	    double nr1=-tab2n.sin(v);
	    double nr2=-tab2n.sin(v+1);
	    double nz1=-tab2n.cos(v);
	    double nz2=-tab2n.cos(v+1);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(u=0;u<nu;u++){
		double r1=rad1+rr1;
		double r2=rad1+rr2;
		double xx=tab1.sin(u);
		double yy=tab1.cos(u);
		double x1=xx*r1;
		double y1=yy*r1;
		double x2=xx*r2;
		double y2=yy*r2;
		glNormal3d(nr1*xx, nr1*yy, nz1);
		glVertex3d(x1, y1, z1);
		glNormal3d(nr2*xx, nr2*yy, nz2);
		glVertex3d(x2, y2, z2);
	    }
	    glEnd();
	}
	break;	
    }
    glPopMatrix();
}

void GeomTriStrip::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    if(verts.size() <= 2)
	return;
    di->polycount+=verts.size()-2;
    switch(di->get_drawtype()){
    case DrawInfoOpenGL::WireFrame:
	{
	    verts[0]->emit_point(di);
	    verts[1]->emit_point(di);
	    for(int i=2;i<verts.size();i++){
		glBegin(GL_LINE_LOOP);
		verts[i-2]->emit_point(di);
		verts[i-1]->emit_point(di);
		verts[i]->emit_point(di);
		glEnd();
	    }
	}
	break;
    case DrawInfoOpenGL::Flat:
	{
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int i=0;i<verts.size();i++){
		verts[i]->emit_point(di);
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	{
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int i=0;i<verts.size();i++){
		verts[i]->emit_all(di);
	    }
	    glEnd();
	}
	break;
    }
}

void GeomVertex::emit_all(DrawInfoOpenGL*)
{
    glVertex3d(p.x(), p.y(), p.z());
}

void GeomVertex::emit_point(DrawInfoOpenGL*)
{
    glVertex3d(p.x(), p.y(), p.z());
}

void GeomVertex::emit_matl(DrawInfoOpenGL*)
{
    // Do nothing
}

void GeomVertex::emit_normal(DrawInfoOpenGL*)
{
    // Do nothing
}

void GeomNVertex::emit_all(DrawInfoOpenGL*)
{
    glNormal3d(normal.x(), normal.y(), normal.z());
    glVertex3d(p.x(), p.y(), p.z());
}

void GeomNVertex::emit_normal(DrawInfoOpenGL*)
{
    glNormal3d(normal.x(), normal.z(), normal.z());
}

void GeomNMVertex::emit_all(DrawInfoOpenGL* di)
{
    di->set_matl(matl.get_rep());
    glNormal3d(normal.x(), normal.y(), normal.z());
    glVertex3d(p.x(), p.y(), p.z());
}

void GeomNMVertex::emit_matl(DrawInfoOpenGL* di)
{
    di->set_matl(matl.get_rep());
}

void GeomMVertex::emit_all(DrawInfoOpenGL* di)
{
    di->set_matl(matl.get_rep());
    glVertex3d(p.x(), p.y(), p.z());
}

void GeomMVertex::emit_matl(DrawInfoOpenGL* di)
{
    di->set_matl(matl.get_rep());
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

