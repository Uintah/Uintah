
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
#include <Geom/Line.h>
#include <Geom/Polyline.h>
#include <Geom/Sphere.h>
#include <Geom/Tetra.h>
#include <Geom/Tri.h>
#include <Geom/TriStrip.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

#include <GL/gl.h>

DrawInfoOpenGL::DrawInfoOpenGL()
: current_matl(0)
{
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
    objdraw(di);
    if(matl.get_rep())
	di->pop_matl();
}

void GeomCone::objdraw(DrawInfoOpenGL* di)
{
    SinCosTable u(nu, 0, 2.*Pi);
    int i,j;
    di->polycount+=nu*nv;
    switch(di->drawtype){
    case DrawInfoOpenGL::WireFrame:
	for(i=0;i<=nv;i++){
	    double z=double(i)/double(nv);
	    double rad=bot_rad+(top_rad-bot_rad)*z;
	    Vector up(axis*z);
	    Point bot_up(bottom+up);
	    glBegin(GL_LINE_LOOP);
	    for(int j=0;j<nu-1;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Vector rv1(v1*(d1*rad));
		Vector rv2(v2*(d2*rad));
		Vector rv(rv1+rv2);
		Point p(bot_up+rv);
		glVertex3d(p.x(), p.y(), p.z());
	    }
	    glEnd();
	}
	glBegin(GL_LINES);
	for(j=0;j<nu-1;j++){
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    Vector trv1(v1*(d1*bot_rad));
	    Vector trv2(v2*(d2*bot_rad));
	    Vector trv(trv1+trv2);
	    Point p1(bottom+trv);
	    Vector brv1(v1*(d1*top_rad));
	    Vector brv2(v2*(d2*top_rad));
	    Vector brv(brv1+brv2);	    
	    Point p2(top+brv);
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	}
	glEnd();
	break;
    case DrawInfoOpenGL::Flat:
	for(i=0;i<nv;i++){
	    double z1=double(i)/double(nv);
	    double z2=double(i+1)/double(nv);
	    double rad1=bot_rad+(top_rad-bot_rad)*z1;
	    double rad2=bot_rad+(top_rad-bot_rad)*z2;
	    Point b1(bottom+axis*z1);
	    Point b2(bottom+axis*z2);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int j=0;j<nu;j++){
		double d1=u.sin(j)*rad1;
		double d2=u.cos(j)*rad1;
		Vector rv1a(v1*d1);
		Vector rv1b(v2*d2);
		Vector rv1(rv1a+rv1b);
		Point p1(b1+rv1);
		double d3=u.sin(j)*rad2;
		double d4=u.cos(j)*rad2;
		Vector rv2a(v1*d3);
		Vector rv2b(v2*d4);
		Vector rv2(rv2a+rv2b);
		Point p2(b2+rv2);
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Gouraud:
	for(j=0;j<nu-1;j++){
	    glBegin(GL_TRIANGLE_STRIP);
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    double d3=u.sin(j+1);
	    double d4=u.cos(j+1);
	    Vector n1(v1*d1+v2*d2);
	    Vector n2(v1*d3+v2*d4);
	    Vector n(n1+axis*tilt);
	    glNormal3d(n.x(), n.y(), n.z());
	    for(i=0;i<=nv;i++){
		double z1=double(i)/double(nv);
		double rad=bot_rad+(top_rad-bot_rad)*z1;
	        Point paz(bottom+axis*z1);
		Point p1(paz+n1*rad);
		Point p2(paz+n2*rad);
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Phong:
	for(i=0;i<nv;i++){
	    double z1=double(i)/double(nv);
	    double z2=double(i+1)/double(nv);
	    double rad1=bot_rad+(top_rad-bot_rad)*z1;
	    double rad2=bot_rad+(top_rad-bot_rad)*z2;
	    Point b1(bottom+axis*z1);
	    Point b2(bottom+axis*z2);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int j=0;j<nu;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Vector n(v1*d1+v2*d2);
		Point p1(b1+n*rad1);
		Point p2(b2+n*rad2);
		Vector nn(n+axis*tilt);
		glNormal3d(-nn.x(), -nn.y(), -nn.z());
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
	    glEnd();
	}
	break;
    }
}

void GeomCylinder::objdraw(DrawInfoOpenGL* di)
{
    SinCosTable u(nu, 0, 2.*Pi);
    int i,j;
    di->polycount+=nu*nv;
    switch(di->drawtype){
    case DrawInfoOpenGL::WireFrame:
	for(i=0;i<=nv;i++){
	    double z=double(i)/double(nv);
	    Vector up(axis*z);
	    Point bot_up(bottom+up);
	    glBegin(GL_LINE_LOOP);
	    for(int j=0;j<nu-1;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Vector rv(v1*d1+v2*d2);
		Point p(bot_up+rv);
		glVertex3d(p.x(), p.y(), p.z());
	    }
	    glEnd();
	}
	glBegin(GL_LINES);
	for(j=0;j<nu-1;j++){
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    Vector rv(v1*d1+v2*d2);
	    Point p1(bottom+rv);
	    Point p2(p1+axis);
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	}
	glEnd();
	break;
    case DrawInfoOpenGL::Flat:
	for(i=0;i<nv;i++){
	    double z1=double(i)/double(nv);
	    double z2=double(i+1)/double(nv);
	    Point b1(bottom+axis*z1);
	    Point b2(bottom+axis*z2);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int j=0;j<nu;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Vector rv(v1*d1+v2*d2);
		Point p1(b1+rv);
		Point p2(b2+rv);
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Gouraud:
	for(j=0;j<nu-1;j++){
	    glBegin(GL_TRIANGLE_STRIP);
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    double d3=u.sin(j+1);
	    double d4=u.cos(j+1);
	    Vector n1(v1*d1+v2*d2);
	    Vector n2(v1*d3+v2*d4);
	    Point pn1(bottom+n1);
	    Point pn2(bottom+n2);
	    glNormal3d(n1.x(), n1.y(), n1.z());
	    for(i=0;i<=nv;i++){
		double z1=double(i)/double(nv);
		Point p1(pn1+axis*z1);
		Point p2(pn2+axis*z1);
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
	    glEnd();
	}
	break;
    case DrawInfoOpenGL::Phong:
	for(i=0;i<nv;i++){
	    double z1=double(i)/double(nv);
	    double z2=double(i+1)/double(nv);
	    Point b1(bottom+axis*z1);
	    Point b2(bottom+axis*z2);
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int j=0;j<nu;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Vector n(v1*d1+v2*d2);
		Point p1(b1+n);
		Point p2(b2+n);
		glNormal3d(-n.x(), -n.y(), -n.z());
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
	    glEnd();
	}
	break;
    }
}

void GeomDisc::objdraw(DrawInfoOpenGL* di)
{
    SinCosTable u(nu, 0, 2.*Pi);
    int i,j;
    di->polycount+=nu*nv;
    switch(di->drawtype){
    case DrawInfoOpenGL::WireFrame:
	for(i=1;i<=nv;i++){
	    glBegin(GL_LINE_LOOP);
	    double r=rad*double(i)/double(nv);
	    for(int j=0;j<nu-1;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Vector rv1a(v1*(d1*r));
		Vector rv1b(v2*(d2*r));
		Vector rv1(rv1a+rv1b);
		Point p(cen+rv1);
		glVertex3d(p.x(), p.y(), p.z());
	    }
	    glEnd();
	}
	glBegin(GL_LINES);
	for(j=0;j<nu-1;j++){
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    Vector rv1a(v1*(d1*rad));
	    Vector rv1b(v2*(d2*rad));
	    Vector rv1(rv1a+rv1b);
	    Point p(cen+rv1);
	    glVertex3d(cen.x(), cen.y(), cen.z());
	    glVertex3d(p.x(), p.y(), p.z());
	}
	glEnd();
	break;
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	glNormal3d(normal.x(), normal.y(), normal.z());
	// Trickle through...
    case DrawInfoOpenGL::Flat:
	for(i=0;i<nv;i++){
	    glBegin(GL_TRIANGLE_STRIP);
	    double r1=rad*double(i)/double(nv);
	    double r2=rad*double(i+1)/double(nv);
	    for(int j=0;j<nu;j++){
		double d1=u.sin(j);
		double d2=u.cos(j);
		Vector rv1a(v1*(d1*r1));
		Vector rv1b(v2*(d2*r1));	
		Vector rv1(rv1a+rv1b);
		Point p1(cen+rv1);
		Vector rv2a(v1*(d1*r2));
		Vector rv2b(v2*(d2*r2));	
		Vector rv2(rv2a+rv2b);
		Point p2(cen+rv2);
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
	    }
	    glEnd();
	}
	break;
    }
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
    cerr << "Drawing polyline...\n";
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
    SinCosTable u(nu, 0, 2.*Pi);
    SinCosTable v(nv, 0, Pi, rad);
    double cx=cen.x();
    double cy=cen.y();
    double cz=cen.z();
    int i, j;
    di->polycount+=nu*nv;
    switch(di->drawtype){
    case DrawInfoOpenGL::WireFrame:
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
    case DrawInfoOpenGL::Flat:
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
    case DrawInfoOpenGL::Gouraud:
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
    case DrawInfoOpenGL::Phong:
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
    switch(di->drawtype){
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
	glNormal3d(n.x(), n.y(), n.z());
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
    switch(di->drawtype){
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

