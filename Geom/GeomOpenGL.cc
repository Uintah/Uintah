
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
#include <Geom/Arrows.h>
#include <Geom/BBoxCache.h>
#include <Geom/Cone.h>
#include <Geom/Cylinder.h>
#include <Geom/Disc.h>
#include <Geom/Geom.h>
#include <Geom/Grid.h>
#include <Geom/Group.h>
#include <Geom/HeadLight.h>
#include <Geom/IndexedGroup.h>
#include <Geom/Light.h>
#include <Geom/Line.h>
#include <Geom/Material.h>
#include <Geom/Pick.h>
#include <Geom/PointLight.h>
#include <Geom/Polyline.h>
#include <Geom/PortManager.h>
#include <Geom/Pt.h>
#include <Geom/RenderMode.h>
#include <Geom/Sphere.h>
#include <Geom/Switch.h>
#include <Geom/Tetra.h>
#include <Geom/Torus.h>
#include <Geom/Tri.h>
#include <Geom/Triangles.h>
#include <Geom/Tube.h>
#include <Geom/TriStrip.h>
#include <Geom/View.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>
#include <Geometry/Plane.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <Geom/Color.h>

#include <Modules/Salmon/Salmon.h>

#include <stdio.h>

#define MAX_MATL_STACK 100

void GeomObj::pre_draw(DrawInfoOpenGL* di, Material* matl, int lit)
{
    if(lit && di->lighting && !di->currently_lit){
	di->currently_lit=1;
	glEnable(GL_LIGHTING);
	switch(di->get_drawtype()) {
	case DrawInfoOpenGL::WireFrame:
	    gluQuadricNormals(di->qobj, GLU_SMOOTH);
	    break;
	case DrawInfoOpenGL::Flat:
	    gluQuadricNormals(di->qobj, GLU_FLAT);
	    break;
	case DrawInfoOpenGL::Gouraud:
	    gluQuadricNormals(di->qobj, GLU_SMOOTH);
	    break;
	}
	
    }
    if((!lit || !di->lighting) && di->currently_lit){
	di->currently_lit=0;
	glDisable(GL_LIGHTING);
	gluQuadricNormals(di->qobj, GLU_NONE);
    }
    di->set_matl(matl);
}

static void quad_error(GLenum code)
{
    cerr << "WARNING: Quadric Error (" << (char*)gluErrorString(code) << ")" << endl;
}

DrawInfoOpenGL::DrawInfoOpenGL()
: current_matl(0),lighting(1),currently_lit(1),pickmode(1),fog(0)
{
    qobj=gluNewQuadric();
    gluQuadricCallback(qobj, GLU_ERROR, (void (*)())quad_error);
}

void DrawInfoOpenGL::reset()
{
    polycount=0;
    current_matl=0;
    ignore_matl=0;
    fog=0;
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
	gluQuadricDrawStyle(qobj, GLU_LINE);
	break;
    case DrawInfoOpenGL::Flat:
	gluQuadricDrawStyle(qobj, GLU_FILL);
	glShadeModel(GL_FLAT);
	break;
    case DrawInfoOpenGL::Gouraud:
	gluQuadricDrawStyle(qobj, GLU_FILL);
	glShadeModel(GL_SMOOTH);
	break;
    }
    
}

void DrawInfoOpenGL::init_lighting(int use_light)
{
    if (use_light) {
	glEnable(GL_LIGHTING);
 	switch(drawtype) {
	case DrawInfoOpenGL::WireFrame:
	    gluQuadricNormals(qobj, GLU_SMOOTH);
	    break;
	case DrawInfoOpenGL::Flat:
	    gluQuadricNormals(qobj, GLU_FLAT);
	    break;
	case DrawInfoOpenGL::Gouraud:
	    gluQuadricNormals(qobj, GLU_SMOOTH);
	    break;
	}
    }
    else {
	glDisable(GL_LIGHTING);
	gluQuadricNormals(qobj,GLU_NONE);
    }
    if (fog)
	glEnable(GL_FOG);
    else
	glDisable(GL_FOG);
}

void DrawInfoOpenGL::set_matl(Material* matl)
{
    if(matl==current_matl || ignore_matl) {
	return;	
    }
    float color[4];
    if (!current_matl) {
	matl->ambient.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
	matl->diffuse.get_color(color);
	glColor4fv(color);
	matl->specular.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
	matl->emission.get_color(color);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, matl->shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, color);
    }    
    else {
	matl->ambient.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);

	matl->diffuse.get_color(color);
	glColor4fv(color);

	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);

	matl->specular.get_color(color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);

	matl->emission.get_color(color);	
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, color);

        if (matl->shininess != current_matl->shininess) {
	    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, matl->shininess);
        }
    }	
    current_matl=matl;
}

// WARNING - doesn''t respond to lighting correctly yet!

void GeomArrows::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    int n=positions.size();
    di->polycount+=6*n;
    // Draw shafts - they are the same for all draw types....
    double shaft_scale=headlength;
    if(di->get_drawtype() == DrawInfoOpenGL::WireFrame)
	shaft_scale=1.0;
    if(shaft_matls.size() == 1){
	pre_draw(di, shaft_matls[0].get_rep(), 0);
	glBegin(GL_LINES);
	for(int i=0;i<n;i++){
	    Point from(positions[i]);
	    Point to(from+directions[i]*shaft_scale);
	    glVertex3d(from.x(), from.y(), from.z());
	    glVertex3d(to.x(), to.y(), to.z());
	}
	glEnd();
    } else {
	pre_draw(di, matl, 0);
	glBegin(GL_LINES);
	for(int i=0;i<n;i++){
	    di->set_matl(shaft_matls[i].get_rep());
	    Point from(positions[i]);
	    Point to(from+directions[i]*shaft_scale);
	    glVertex3d(from.x(), from.y(), from.z());
	    glVertex3d(to.x(), to.y(), to.z());
	}
	glEnd();
    }

    // Draw back and head
    switch(di->get_drawtype()){
    case DrawInfoOpenGL::WireFrame:
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	break;
    }

    int do_normals=1;
    if(di->get_drawtype() == DrawInfoOpenGL::Flat ||
       di->get_drawtype() == DrawInfoOpenGL::WireFrame)
	do_normals=0;
    if(back_matls.size() == 1){
	pre_draw(di, back_matls[0].get_rep(), 1);
	glBegin(GL_QUADS);
	if(do_normals){
	    for(int i=0;i<n;i++){
		Point from(positions[i]+directions[i]*headlength);
		glNormal3d(directions[i].x(), directions[i].y(), directions[i].z());
		Point to(from+directions[i]);
		Point p1(from+v1[i]);
		glVertex3d(p1.x(), p1.y(), p1.z());
		Point p2(from+v2[i]);
		glVertex3d(p2.x(), p2.y(), p2.z());
		Point p3(from-v1[i]);
		glVertex3d(p3.x(), p3.y(), p3.z());
		Point p4(from-v2[i]);
		glVertex3d(p4.x(), p4.y(), p4.z());
	    }
	} else {
	    for(int i=0;i<n;i++){
		Point from(positions[i]+directions[i]*headlength);
		Point to(from+directions[i]);
		Vector n1(v1[i]+v2[i]);
		Point p1(from+v1[i]);
		glVertex3d(p1.x(), p1.y(), p1.z());
		Point p2(from+v2[i]);
		glVertex3d(p2.x(), p2.y(), p2.z());
		Point p3(from-v1[i]);
		glVertex3d(p3.x(), p3.y(), p3.z());
		Point p4(from-v2[i]);
		glVertex3d(p4.x(), p4.y(), p4.z());
	    }
	}
	glEnd();
    } else {
	pre_draw(di, matl, 1);
	glBegin(GL_QUADS);
	if(do_normals){
	    for(int i=0;i<n;i++){
		di->set_matl(back_matls[i].get_rep());
		glNormal3d(directions[i].x(), directions[i].y(), directions[i].z());
		Point from(positions[i]+directions[i]*headlength);
		Point to(from+directions[i]);
		Point p1(from+v1[i]);
		glVertex3d(p1.x(), p1.y(), p1.z());
		Point p2(from+v2[i]);
		glVertex3d(p2.x(), p2.y(), p2.z());
		Point p3(from-v1[i]);
		glVertex3d(p3.x(), p3.y(), p3.z());
		Point p4(from-v2[i]);
		glVertex3d(p4.x(), p4.y(), p4.z());
	    }
	} else {
	    for(int i=0;i<n;i++){
		di->set_matl(back_matls[i].get_rep());
		Point from(positions[i]+directions[i]*headlength);
		Point to(from+directions[i]);
		Point p1(from+v1[i]);
		glVertex3d(p1.x(), p1.y(), p1.z());
		Point p2(from+v2[i]);
		glVertex3d(p2.x(), p2.y(), p2.z());
		Point p3(from-v1[i]);
		glVertex3d(p3.x(), p3.y(), p3.z());
		Point p4(from-v2[i]);
		glVertex3d(p4.x(), p4.y(), p4.z());
	    }
	}
	glEnd();
    }
    if(head_matls.size() == 1){
	pre_draw(di, head_matls[0].get_rep(), 1);
	if(do_normals){
	    double w=headwidth;
	    double h=1.0-headlength;
	    double w2h2=w*w/h;
	    for(int i=0;i<n;i++){
		glBegin(GL_TRIANGLES);
		Vector dn(directions[i]*w2h2);
		Vector n(dn+v1[i]+v2[i]);
		glNormal3d(n.x(), n.y(), n.z());

		Point top(positions[i]+directions[i]);
		Point from=top-directions[i]*h;
		Point to(from+directions[i]);
		Point p1(from+v1[i]);
		Point p2(from+v2[i]);
		glVertex3d(top.x(), top.y(), top.z());
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z()); // 1st tri
		n=dn-v1[i]+v2[i];
		glNormal3d(n.x(), n.y(), n.z());
		Point p3(from-v1[i]);
		glVertex3d(top.x(), top.y(), top.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
		glVertex3d(p3.x(), p3.y(), p3.z()); // 2nd tri
		n=dn-v1[i]-v2[i];
		glNormal3d(n.x(), n.y(), n.z());
		Point p4(from-v2[i]);
		glVertex3d(top.x(), top.y(), top.z());
		glVertex3d(p3.x(), p3.y(), p3.z());
		glVertex3d(p4.x(), p4.y(), p4.z()); // 3rd tri
		n=dn+v1[i]-v2[i];
		glNormal3d(n.x(), n.y(), n.z());
		glVertex3d(top.x(), top.y(), top.z());
		glVertex3d(p4.x(), p4.y(), p4.z());
		glVertex3d(p1.x(), p1.y(), p1.z()); // 4th tri
		glEnd();
	    }
	} else {
	    for(int i=0;i<n;i++){
		glBegin(GL_TRIANGLE_FAN);
		Point from(positions[i]+directions[i]);
		glVertex3d(from.x(), from.y(), from.z());
		from-=directions[i]*(1.0-headlength);
		Point to(from+directions[i]);
		Point p1(from+v1[i]);
		glVertex3d(p1.x(), p1.y(), p1.z());
		Point p2(from+v2[i]);
		glVertex3d(p2.x(), p2.y(), p2.z());
		Point p3(from-v1[i]);
		glVertex3d(p3.x(), p3.y(), p3.z());
		Point p4(from-v2[i]);
		glVertex3d(p4.x(), p4.y(), p4.z());
		glVertex3d(p1.x(), p1.y(), p1.z());
		glEnd();
	    }
	}
    } else {
	pre_draw(di, matl, 1);
	if(do_normals){
	    double w=headwidth;
	    double h=1.0-headlength;
	    double w2h2=w*w+h*h;
	    for(int i=0;i<n;i++){
		glBegin(GL_TRIANGLES);
		Vector dn(directions[i]*w2h2);
		Vector n(dn+v1[i]+v2[i]);
		glNormal3d(n.x(), n.y(), n.z());
		di->set_matl(back_matls[i].get_rep());

		Point top(positions[i]+directions[i]);
		Point from=top-directions[i]*h;
		Point to(from+directions[i]);
		Point p1(from+v1[i]);
		Point p2(from+v2[i]);
		glVertex3d(top.x(), top.y(), top.z());
		glVertex3d(p1.x(), p1.y(), p1.z());
		glVertex3d(p2.x(), p2.y(), p2.z()); // 1st tri
		n=dn-v1[i]+v2[i];
		glNormal3d(n.x(), n.y(), n.z());
		Point p3(from-v1[i]);
		glVertex3d(top.x(), top.y(), top.z());
		glVertex3d(p2.x(), p2.y(), p2.z());
		glVertex3d(p3.x(), p3.y(), p3.z()); // 2nd tri
		n=dn-v1[i]-v2[i];
		glNormal3d(n.x(), n.y(), n.z());
		Point p4(from-v2[i]);
		glVertex3d(top.x(), top.y(), top.z());
		glVertex3d(p3.x(), p3.y(), p3.z());
		glVertex3d(p4.x(), p4.y(), p4.z()); // 3rd tri
		n=dn+v1[i]-v2[i];
		glNormal3d(n.x(), n.y(), n.z());
		glVertex3d(top.x(), top.y(), top.z());
		glVertex3d(p4.x(), p4.y(), p4.z());
		glVertex3d(p1.x(), p1.y(), p1.z()); // 4th tri
		glEnd();
	    }
	} else {
	    for(int i=0;i<n;i++){
		glBegin(GL_TRIANGLE_FAN);
		di->set_matl(back_matls[i].get_rep());
		Point from(positions[i]+directions[i]);
		glVertex3d(from.x(), from.y(), from.z());
		from-=directions[i]*(1.0-headlength);
		Point to(from+directions[i]);
		Point p1(from+v1[i]);
		glVertex3d(p1.x(), p1.y(), p1.z());
		Point p2(from+v2[i]);
		glVertex3d(p2.x(), p2.y(), p2.z());
		Point p3(from-v1[i]);
		glVertex3d(p3.x(), p3.y(), p3.z());
		Point p4(from-v2[i]);
		glVertex3d(p4.x(), p4.y(), p4.z());
		glVertex3d(p1.x(), p1.y(), p1.z());
		glEnd();
	    }
	}
    }

    switch(di->get_drawtype()){
    case DrawInfoOpenGL::WireFrame:
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	break;
    }
}

void GeomBBoxCache::draw(DrawInfoOpenGL* di, Material *m, double time)
{
    child->draw(di,m,time);
}

void GeomCappedCylinder::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    if(height < 1.e-6 || rad < 1.e-6)return;
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
    if(height < 1.e-6 || (bot_rad < 1.e-6 && top_rad < 1.e-6))return;
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
    if(height < 1.e-6 || (bot_rad < 1.e-6 && top_rad < 1.e-6))return;
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(bottom.x(), bottom.y(), bottom.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=2*(nu-1)*(nv-1);
    gluCylinder(di->qobj, bot_rad, top_rad, height, nu, nv);
    if(bot_rad > 1.e-6){
        // Bottom endcap
        di->polycount+=2*(nu-1)*(nvdisc1-1);
        gluDisk(di->qobj, 0, bot_rad, nu, nvdisc1);
    }
    if(top_rad > 1.e-6){
        // Top endcap
        glTranslated(0, 0, height);
        di->polycount+=2*(nu-1)*(nvdisc2-1);
        gluDisk(di->qobj, 0, top_rad, nu, nvdisc2);
    }
    glPopMatrix();
}

void GeomContainer::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
    child->draw(di, matl, time);
}

void GeomCylinder::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    if(height < 1.e-6 || rad < 1.e-6)return;
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
    if(rad < 1.e-6)return;
    pre_draw(di, matl, 1);
    glPushMatrix();
    glTranslated(cen.x(), cen.y(), cen.z());
    glRotated(RtoD(zrotangle), zrotaxis.x(), zrotaxis.y(), zrotaxis.z());
    di->polycount+=2*(nu-1)*(nv-1);
    gluDisk(di->qobj, 0, rad, nu, nv);
    glPopMatrix();
}

// WARNING doesn't respond to lighting correctly yet!

void GeomGrid::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    int nu=verts.dim1();
    int nv=verts.dim2();
    di->polycount+=2*(nu-1)*(nv-1);
    Vector uu(u/(nu-1));
    Vector vv(v/(nv-1));

    switch(di->get_drawtype()){
    case DrawInfoOpenGL::WireFrame:
	{
	    Point rstart(corner);
	    for(int i=0;i<nu;i++){
		Point p1(rstart);
		glBegin(GL_LINE_STRIP);
		for(int j=0;j<nv;j++){
		    Point pp1(p1+w*verts(i, j));
		    if(have_matls)
			di->set_matl(matls(i, j).get_rep());
		    if(have_normals){
			Vector normal(normals(i, j));
			glNormal3d(normal.x(), normal.y(), normal.z());
		    }
		    glVertex3d(pp1.x(), pp1.y(), pp1.z());

		    p1+=vv;
		}
		glEnd();
		rstart+=uu;
	    }
	    rstart=corner;
	    for(int j=0;j<nv;j++){
		Point p1(rstart);
		glBegin(GL_LINE_STRIP);
		for(int i=0;i<nu;i++){
		    Point pp1(p1+w*verts(i, j));
		    if(have_matls)
			di->set_matl(matls(i, j).get_rep());
		    if(have_normals){
			Vector normal(normals(i, j));
			glNormal3d(normal.x(), normal.y(), normal.z());
		    }
		    glVertex3d(pp1.x(), pp1.y(), pp1.z());

		    p1+=uu;
		}
		glEnd();
		rstart+=vv;
	    }
	}
	break;
    case DrawInfoOpenGL::Flat:
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	{
	    if(!have_normals)
		glNormal3d(w.x(), w.y(), w.z());
	    Point rstart(corner);
	    for(int i=0;i<nu-1;i++){
		Point p1(rstart);
		Point p2(rstart+uu);
		rstart=p2;
		glBegin(GL_TRIANGLE_STRIP);
		for(int j=0;j<nv;j++){
		    Point pp1(p1+w*verts(i, j));
		    Point pp2(p2+w*verts(i+1, j));
		    if(have_matls)
			di->set_matl(matls(i, j).get_rep());
		    if(have_normals){
			Vector normal(normals(i, j));
			glNormal3d(normal.x(), normal.y(), normal.z());
		    }
		    glVertex3d(pp1.x(), pp1.y(), pp1.z());

		    if(have_matls)
			di->set_matl(matls(i+1, j).get_rep());
		    if(have_normals){
			Vector normal(normals(i+1, j));
			glNormal3d(normal.x(), normal.y(), normal.z());
		    }
		    glVertex3d(pp2.x(), pp2.y(), pp2.z());
		    p1+=vv;
		    p2+=vv;
		}
		glEnd();
	    }
	}
	break;
    }
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
    if(di->pickmode){
#if (_MIPS_SZPTR == 64)
	unsigned long o=(unsigned long)this;
	unsigned int o1=(o>>32)&0xffffffff;
	unsigned int o2=o&0xffffffff;
	glPushName(o1);
	glPushName(o2);
	cerr << "pick: " << this << endl;
#else
	glPushName((GLuint)this);
#endif
    }
    if(selected && highlight.get_rep()){
	di->set_matl(highlight.get_rep());
	int old_ignore=di->ignore_matl;
	di->ignore_matl=1;
	child->draw(di, highlight.get_rep(), time);
	di->ignore_matl=old_ignore;
    } else {
	child->draw(di, matl, time);
    }
    if(di->pickmode){
#if (_MIPS_SZPTR == 64)
	glPopName();
	glPopName();
#else
	glPopName();
#endif
    }
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

// WARNING not fixed for lighting yet!

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
		int j;
		for(j=0;j<nu;j++){
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
  if(rad < 1.e-6)return;
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
	if (di->currently_lit) {
	    Vector n1(Plane(p1, p2, p3).normal());
	    Vector n2(Plane(p1, p2, p4).normal());
	    Vector n3(Plane(p4, p2, p3).normal());
	    Vector n4(Plane(p1, p4, p3).normal());
	    glBegin(GL_LINE_STRIP);
	    glNormal3d(n1.x(),n1.y(),n1.z());
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
	else {
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
	break;
    case DrawInfoOpenGL::Flat:
	// this should be made into a tri-strip, but I couldn;t remember how...

	/* this can be done as a triangle strip using 8 vertices, or
	 * as a triangle fan with 5 and 1 single triangle (8 verts)
	 * I am doing the fan now (ordering is wierd with a tri-strip), but
	 * will switch to the tri-strip when I can test it, if it's faster
	 */	
    case DrawInfoOpenGL::Gouraud:
    case DrawInfoOpenGL::Phong:
	// this should be made into a tri-strip, but I couldn;t remember how...

	/*
	 * These are actualy just flat shaded, to get "gourad" shading
	 * you could average the facet normals for all the faces touching
	 * a given vertex.  I don't think there is a faster way to do this
	 * using flat shading.
	 */
	if (di->currently_lit) {
	    Vector n1(Plane(p1, p2, p3).normal());
	    Vector n2(Plane(p1, p2, p4).normal());
	    Vector n3(Plane(p4, p2, p3).normal());
	    Vector n4(Plane(p1, p4, p3).normal());

	    glBegin(GL_TRIANGLES);
	    glNormal3d(n1.x(), n1.y(), n1.z());
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	 
	    glNormal3d(n2.x(), n2.y(), n2.z());
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());

	    glNormal3d(n3.x(), n3.y(), n3.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());

	    glNormal3d(n4.x(), n4.y(), n4.z());
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glEnd();
	}
	else {
	    glBegin(GL_TRIANGLE_FAN);
	    glVertex3d(p1.x(), p1.y(), p1.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glEnd();
	    glBegin(GL_TRIANGLES);
	    glVertex3d(p4.x(), p4.y(), p4.z());
	    glVertex3d(p2.x(), p2.y(), p2.z());
	    glVertex3d(p3.x(), p3.y(), p3.z());
	    glEnd();
	}
#if 0
	/*
	 * This has inconsistant ordering....
	 */ 
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
#endif
	break;
    }
}

void GeomTimeSwitch::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
    if(time >= tbeg && time < tend)
	child->draw(di, matl, time);
}

// WARNING not fixed for lighting correctly yet!

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
    if (di->currently_lit) {
	switch(di->get_drawtype()){
	case DrawInfoOpenGL::WireFrame:
	    glBegin(GL_LINE_LOOP);
	    verts[0]->emit_all(di);
	    verts[1]->emit_all(di);
	    verts[2]->emit_all(di);
	    glEnd();
	    break;
	case DrawInfoOpenGL::Flat:
	    glBegin(GL_TRIANGLES);
	    glNormal3d(-n.x(), -n.y(), -n.z());
	    verts[0]->emit_point(di);
	    verts[1]->emit_point(di);
	    verts[2]->emit_all(di);
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
    else {
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
	    verts[2]->emit_matl(di);
	    verts[2]->emit_point(di);
	    glEnd();
	    break;
	case DrawInfoOpenGL::Gouraud:
	case DrawInfoOpenGL::Phong:
	    // posible change to just material and point...
	    glBegin(GL_TRIANGLES);
	    verts[0]->emit_all(di);
	    verts[1]->emit_all(di);
	    verts[2]->emit_all(di);
	    glEnd();
	    break;
	}
    }
}

void GeomTriangles::draw(DrawInfoOpenGL* di, Material* matl, double)
{
    pre_draw(di, matl, 1);
    if(verts.size() <= 2)
	return;
    di->polycount+=verts.size()/3;
    if (di->currently_lit) {
	glDisable(GL_NORMALIZE);
	switch(di->get_drawtype()){
	case DrawInfoOpenGL::WireFrame:
	    {
		for(int i=0;i<verts.size();i+=3){
		    glBegin(GL_LINE_LOOP);
		    glNormal3d(normals[i/3].x(), normals[i/3].y(), 
			       normals[i/3].z());
		    verts[i]->emit_all(di);
		    verts[i+1]->emit_all(di);
		    verts[i+2]->emit_all(di);
		    glEnd();
		}
	    }
	    break;
	case DrawInfoOpenGL::Flat:
	    {
		glBegin(GL_TRIANGLES);
		for(int i=0;i<verts.size();i+=3){
		    glNormal3d(normals[i/3].x(), normals[i/3].y(), 
			       normals[i/3].z());
		    verts[i]->emit_point(di);
		    verts[i+1]->emit_point(di);
		    verts[i+2]->emit_all(di);
		}
		glEnd();
	    }
	    break;
	case DrawInfoOpenGL::Gouraud:
	case DrawInfoOpenGL::Phong:
	    {
		glBegin(GL_TRIANGLES);
		for(int i=0;i<verts.size();i+=3){
		    glNormal3d(normals[i/3].x(), normals[i/3].y(), 
			       normals[i/3].z());
		    verts[i]->emit_all(di);
		    verts[i+1]->emit_all(di);
		    verts[i+2]->emit_all(di);
		}
		glEnd();
	    }
	    break;
	}
	glEnable(GL_NORMALIZE);
    }
    else {
	switch(di->get_drawtype()){
	case DrawInfoOpenGL::WireFrame:
	    {
		for(int i=0;i<verts.size();i+=3){
		    glBegin(GL_LINE_LOOP);
		    verts[i]->emit_all(di);
		    verts[i+1]->emit_all(di);
		    verts[i+2]->emit_all(di);
		    glEnd();
		}
	    }
	    break;
	case DrawInfoOpenGL::Flat:
	    {
		glBegin(GL_TRIANGLES);
		for(int i=0;i<verts.size();i+=3){
		    verts[i]->emit_point(di);
		    verts[i+1]->emit_point(di);
		    verts[i+2]->emit_all(di);
		}
		glEnd();
	    }
	    break;
	case DrawInfoOpenGL::Gouraud:
	case DrawInfoOpenGL::Phong:
	    {
		glDisable(GL_NORMALIZE);
		glBegin(GL_TRIANGLES);
		for(int i=0;i<verts.size();i+=3){
		    verts[i]->emit_all(di);
		    verts[i+1]->emit_all(di);
		    verts[i+2]->emit_all(di);
		}
		glEnd();
		glEnable(GL_NORMALIZE);
	    }
	    break;
	}
    }
}

// WARNING not fixed for lighting correctly yet!

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
	{
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
    if (di->currently_lit) {
	glDisable(GL_NORMALIZE);
	switch(di->get_drawtype()){
	case DrawInfoOpenGL::WireFrame:
	    {
		verts[0]->emit_all(di);
		verts[1]->emit_all(di);
		for(int i=2;i<verts.size();i++){
		    glBegin(GL_LINE_LOOP);
		    verts[i-2]->emit_all(di);
		    verts[i-1]->emit_all(di);
		    verts[i]->emit_all(di);
		    glEnd();
		}
	    }
	    break;
	case DrawInfoOpenGL::Flat:
	    {
		glBegin(GL_TRIANGLE_STRIP);
		for(int i=0;i<verts.size();i++){
		    verts[i]->emit_all(di);
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
	glEnable(GL_NORMALIZE);
    }
    else {
	switch(di->get_drawtype()){
	case DrawInfoOpenGL::WireFrame:
	    {
		verts[0]->emit_matl(di);
		verts[0]->emit_point(di);
		verts[1]->emit_matl(di);
		verts[1]->emit_point(di);
		for(int i=2;i<verts.size();i++){
		    glBegin(GL_LINE_LOOP);
		    verts[i-2]->emit_matl(di);
		    verts[i-2]->emit_point(di);
		    verts[i-1]->emit_matl(di);
		    verts[i-1]->emit_point(di);
		    verts[i]->emit_matl(di);
		    verts[i]->emit_point(di);
		    glEnd();
		}
	    }
	    break;
	case DrawInfoOpenGL::Flat:
	    {
		glBegin(GL_TRIANGLE_STRIP);
		for(int i=0;i<verts.size();i++){
		    verts[i]->emit_matl(di);
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
		    verts[i]->emit_matl(di);
		    verts[i]->emit_point(di);
		}
		glEnd();
	    }
	    break;
	}
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

void GeomCVertex::emit_all(DrawInfoOpenGL* /*di*/)
{
    glColor3d(color.r(),color.g(),color.b());
    glVertex3d(p.x(), p.y(), p.z());
}

void GeomCVertex::emit_matl(DrawInfoOpenGL* /*di*/)
{
    glColor3d(color.r(),color.g(),color.b());
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
    Point p(view.eyep());
    float f[4];
    f[0]=p.x(); f[1]=p.y(); f[2]=p.z(); f[3]=1.0;
    glLightfv(GL_LIGHT0+i, GL_POSITION, f);
    c.get_color(f);
    glLightfv(GL_LIGHT0+i, GL_DIFFUSE, f);
    glLightfv(GL_LIGHT0+i, GL_SPECULAR, f);
}

void GeomIndexedGroup::draw(DrawInfoOpenGL* di, Material* m, double time)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->draw(di,m,time);
    }
    
}
