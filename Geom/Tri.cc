
/*
 *  Tri.cc: Triangles...
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Tri.h>
#include <Classlib/NotFinished.h>
#include <Geom/GeomRaytracer.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Geometry/Ray.h>
#include <Math/MinMax.h>
#include <iostream.h>

GeomTri::GeomTri(const Point& p1, const Point& p2, const Point& p3)
: GeomObj(), p1(p1), p2(p2), p3(p3), n(Cross(p3-p1, p2-p1))
{
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
    }
}

GeomTri::GeomTri(const GeomTri &copy)
: GeomObj(copy), p1(copy.p1), p2(copy.p2), p3(copy.p3), n(copy.n)
{
}

GeomTri::~GeomTri()
{
}

GeomObj* GeomTri::clone()
{
    return new GeomTri(*this);
}

void GeomTri::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
}

void GeomTri::get_bounds(BSphere& bs)
{
    Vector e1(p2-p1);
    Vector e2(p3-p2);
    Vector e3(p1-p3);
    double e1l2=e1.length2();
    double e2l2=e2.length2();
    double e3l2=e3.length2();
    Point cen;
    double rad;
    int do_circum=0;
    if(e1l2 > e2l2 && e1l2 > e3l2){
	cen=Interpolate(p1, p2, 0.5);
	double rad2=e1l2/4.;
	double dist2=(p3-cen).length2();
	if(dist2 > rad2){
	    do_circum=1;
	} else {
	    rad=Sqrt(rad2);
	}
    } else if(e2l2 > e1l2 && e2l2 > e3l2){
	cen=Interpolate(p2, p3, 0.5);
	double rad2=e2l2/4.;
	double dist2=(p1-cen).length2();
	if(dist2 > rad2){
	    do_circum=1;
	} else {
	    rad=Sqrt(rad2);
	}
    } else {
	cen=Interpolate(p3, p1, 0.5);
	double rad2=e3l2/4.;
	double dist2=(p2-cen).length2();
	if(dist2 > rad2){
	    do_circum=1;
	} else {
	    rad=Sqrt(rad2);
	}
    }
    if(do_circum){
	double d1=-Dot(e3, e1);
	double d2=-Dot(e2, e1);
	double d3=-Dot(e3, e2);
	double c1=d2*d3;
	double c2=d3*d1;
	double c3=d1*d2;
	double c=c1+c2+c3;
	double r2circ=0.25*(d1+d2)*(d2+d3)*(d3+d1)/c;

	// Use the circumcircle radius...
	cen=AffineCombination(p1, (c2+c3)/(2*c),
			      p2, (c3+c1)/(2*c),
			      p3, (c1+c2)/(2*c));
	rad=Sqrt(r2circ);
    }
    ASSERT((cen-p1).length() <= rad*1.00001);
    ASSERT((cen-p2).length() <= rad*1.00001);
    ASSERT((cen-p3).length() <= rad*1.00001);
    bs.extend(cen, rad);
}

void GeomTri::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>& dontfree)
{
    GeomTri* tri=this;
    dontfree.add(tri);
}

void GeomTri::preprocess()
{
    // Nothing to be done...
}

void GeomTri::intersect(const Ray& ray, Material* matl, Hit& hit)
{
    double tmp=Dot(n, ray.direction());
    if(tmp > -1.e-6 && tmp < 1.e-6)return; // Parallel to plane
    Vector v=p1-ray.origin();
    double t=Dot(n, v)/tmp;
    if(t<1.e-6)return;
    if(hit.hit() && t > hit.t())return;
    Point p=ray.origin()+ray.direction()*t;
    double pp1[2], pp2[2], pp3[2] , pt[2];
    double nx=Abs(n.x());
    double ny=Abs(n.y());
    double nz=Abs(n.z());
    if(nx > ny && nx > nz){
	pp1[0]=p1.y(); pp1[1]=p1.z();
	pp2[0]=p2.y(); pp2[1]=p2.z();
	pp3[0]=p3.y(); pp3[1]=p3.z();
	pt[0]=p.y(); pt[1]=p.z();
    } else if(ny > nz && ny > nx){
	pp1[0]=p1.z(); pp1[1]=p1.x();
	pp2[0]=p2.z(); pp2[1]=p2.x();
	pp3[0]=p3.z(); pp3[1]=p3.x();
	pt[0]=p.z(); pt[1]=p.x();
    } else {
	pp1[0]=p1.x(); pp1[1]=p1.y();
	pp2[0]=p2.x(); pp2[1]=p2.y();
	pp3[0]=p3.x(); pp3[1]=p3.y();
	pt[0]=p.x(); pt[1]=p.y();
    }

    int nc=x_cross(pp1, pp2, pt);
    nc+=x_cross(pp2, pp3, pt);
    if(nc==2)return;
    nc+=x_cross(pp3, pp1, pt);
    if(nc==1){
	// We hit!!!
	hit.hit(t, this, matl);
    }
}

int GeomTri::x_cross(double p1[2], double p2[2], double p[2])
{
    // Cut off left
    if(p[0] < p1[0] && p[0] <= p2[0])
	return 0;
    // Cut off botton
    if(p[1] < p1[1] && p[1] <= p2[1])
	return 0;
    // Cut off top
    if(p[1] > p1[1] && p[1] >= p2[1])
	return 0;
    // If on right, then it definitely crosses
    if(p[0] > p1[0] && p[0] >= p2[0])
	return 1;
    // General case...
    double delta=p2[1]-p1[1];
    if(delta < 1.e-6 && delta > -1.e-6)
	return 0;
    double yt=p[1]-p1[1];
    double x=yt/delta*(p2[0]-p1[0])+p1[0];
    if(p[0] < x)
	return 0; // p is left of edge
    else
	return 1; // p is right of edge
}

Vector GeomTri::normal(const Point&, const Hit&)
{
    return n;
}
