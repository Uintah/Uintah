
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
#include <Geom/GeomRaytracer.h>
#include <Geometry/BBox.h>
#include <Geometry/Ray.h>
#include <Math/MinMax.h>
#include <iostream.h>

GeomTri::GeomTri(const Point& p1, const Point& p2, const Point& p3)
: GeomObj(1), p1(p1), p2(p2), p3(p3), n(Cross(p3-p1, p2-p1))
{
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
    }
}

GeomTri::GeomTri(const GeomTri &copy)
: GeomObj(1), p1(copy.p1), p2(copy.p2), p3(copy.p3), n(copy.n)
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

void GeomTri::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>& dontfree)
{
    GeomTri* tri=this;
    dontfree.add(tri);
}

void GeomTri::intersect(const Ray& ray, Material* matl, Hit& hit)
{
    double tmp=Dot(n, ray.direction());
    if(tmp < 1.e-6 || tmp > 1.e-6)return; // Parallel to plane
    Vector v=p1-ray.origin();
    double t=Dot(n, v)/tmp;
    if(t<1.e-6)return;
    if(hit.hit() && t > hit.t())return;
    Point p=ray.origin()+ray.direction()*t;
    double pp1[2], pp2[2], pp3[2] , pt[2];
    if(n.x() > n.y() && n.x() > n.z()){
	pp1[0]=p1.y(); pp1[1]=p1.z();
	pp2[0]=p2.y(); pp2[1]=p2.z();
	pp3[0]=p3.y(); pp3[1]=p3.z();
	pt[0]=p.y(); pt[1]=p.z();
    } else if(n.y() > n.z() && n.y() > n.z()){
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
    if(delta < 1.e-6)
	return 0;
    double yt=p[1]-p1[1];
    double x=yt/delta*(p2[0]-p1[0])+p1[0];
    if(p[0] < x)
	return 0; // p is left of edge
    else
	return 1; // p is right of edge
}
