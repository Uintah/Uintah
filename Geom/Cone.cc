
/*
 *  Cone.h: Cone object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Cone.h>
#include <Classlib/NotFinished.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

GeomCone::GeomCone(int nu, int nv)
: GeomObj(1), nu(nu), nv(nv), bottom(0,0,0), top(0,0,1), bot_rad(1), top_rad(0)
{
}

GeomCone::GeomCone(const Point& bottom, const Point& top,
		   double bot_rad, double top_rad, int nu, int nv)
: GeomObj(1), bottom(bottom), top(top), bot_rad(bot_rad),
  top_rad(top_rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomCone::move(const Point& _bottom, const Point& _top,
		    double _bot_rad, double _top_rad, int _nu, int _nv)
{
    bottom=_bottom;
    top=_top;
    bot_rad=_bot_rad;
    top_rad=_top_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomCone::GeomCone(const GeomCone& copy)
: GeomObj(1), v1(copy.v1), v2(copy.v2), bottom(copy.bottom), top(copy.top),
  bot_rad(copy.bot_rad), top_rad(copy.top_rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomCone::~GeomCone()
{
}

GeomObj* GeomCone::clone()
{
    return new GeomCone(*this);
}

void GeomCone::adjust()
{
    axis=top-bottom;
    height=axis.length();
    if(height < 1.e-6){
	cerr << "Degenerate Cone!\n";
    } else {
	axis.find_orthogonal(v1, v2);
    }
    tilt=(bot_rad-top_rad)/axis.length2();
    Vector z(0,0,1);	
    if(Abs(axis.y()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(axis, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, axis)/height;
    zrotangle=-Acos(cangle);
}

void GeomCone::get_bounds(BBox& bb)
{
    NOT_FINISHED("GeomCone::get_bounds");
    bb.extend(bottom, bot_rad);
    bb.extend(top, top_rad);
}

void GeomCone::get_bounds(BSphere&)
{
    NOT_FINISHED("GeomCone::get_bounds");
}

void GeomCone::make_prims(Array1<GeomObj*>& free,
			  Array1<GeomObj*>&)
{
    SinCosTable u(nu, 0, 2.*Pi);
    for(int i=0;i<nv;i++){
	double z1=double(i)/double(nv);
	double z2=double(i+1)/double(nv);
	double rad1=bot_rad+(top_rad-bot_rad)*z1;
	double rad2=bot_rad+(top_rad-bot_rad)*z2;
	Point b1(bottom+axis*z1);
	Point b2(bottom+axis*z2);
	Point l1, l2;
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
	    if(j>0){
		GeomTri* t1=new GeomTri(l1, l2, p1);
//		t1->set_matl(matl);
		free.add(t1);
		GeomTri* t2=new GeomTri(l2, p1, p2);
//		t2->set_matl(matl);
		free.add(t2);
	    }
	    l1=p1;
	    l2=p2;
	}
    }
}

void GeomCone::preprocess()
{
    NOT_FINISHED("GeomCone::preprocess");
}

void GeomCone::intersect(const Ray&, Material*,
			 Hit&)
{
    NOT_FINISHED("GeomCone::intersect");
}

