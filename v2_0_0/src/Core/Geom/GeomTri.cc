/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Geom/GeomTri.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomTri()
{
    return scinew GeomTri(Point(0,0,0), Point(1,0,0), Point(0,1,0));
}

PersistentTypeID GeomTri::type_id("GeomTri", "GeomObj", make_GeomTri);

GeomTri::GeomTri(const Point& p1, const Point& p2, const Point& p3)
: n(Cross(p3-p1, p2-p1))
{
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
    }
    add(p1);
    add(p2);
    add(p3);
}

GeomTri::GeomTri(const Point& p1, const Point& p2, const Point& p3,
		 const MaterialHandle& mat1, const MaterialHandle& mat2,
		 const MaterialHandle& mat3)
: n(Cross(p3-p1, p2-p1))
{
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
    }
    add(p1, mat1);
    add(p2, mat2);
    add(p3, mat3);
}

GeomTri::GeomTri(const GeomTri &copy)
: GeomVertexPrim(copy), n(copy.n)
{
}

GeomTri::~GeomTri()
{
}

GeomObj* GeomTri::clone()
{
    return scinew GeomTri(*this);
}

#ifdef BSPHERE
void GeomTri::get_bounds(BSphere& bs)
{
    Point p1(verts[0]->p);
    Point p2(verts[1]->p);
    Point p3(verts[2]->p);
    Vector e1(p2-p1);
    Vector e2(p3-p2);
    Vector e3(p1-p3);
    double e1l2=e1.length2();
    double e2l2=e2.length2();
    double e3l2=e3.length2();
    Point cen;
    double rad=0;
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
#endif

#define GEOMTRI_VERSION 1

void GeomTri::io(Piostream& stream)
{
    stream.begin_class("GeomTri", GEOMTRI_VERSION);
    GeomVertexPrim::io(stream);
    stream.end_class();
}

} // End namespace SCIRun

