//static char *id="@(#) $Id$";

/*
 *  Plane.cc: Uniform Plane containing triangular elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geometry/Plane.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

namespace SCICore {
namespace Geometry {

Plane::Plane()
: n(Vector(0,0,1)), d(0)
{
}

Plane::Plane(double a, double b, double c, double d) : n(Vector(a,b,c)), d(d) {
    double l=n.length();
    d/=l;
    n.normalize();
}

Plane::Plane(const Plane &copy)
: n(copy.n), d(copy.d)
{
}

Plane::Plane(const Point &p1, const Point &p2, const Point &p3) {
    Vector v1(p2-p1);
    Vector v2(p2-p3);
    n=Cross(v2,v1);
    n.normalize();
    d=-Dot(p1, n);
}

Plane::~Plane() {
}

void Plane::flip() {
   n.x(-n.x());
   n.y(-n.y());
   n.z(-n.z());
}

double Plane::eval_point(const Point &p) const {
    return Dot(p, n)+d;
}

Point Plane::project(const Point& p) const
{
   return p-n*(d+Dot(p,n));
}

Vector Plane::project(const Vector& v) const
{
    return v-n*Dot(v,n);
}

Vector Plane::normal() const
{
   return n;
}

void
Plane::ChangePlane(const Point &p1, const Point &p2, const Point &p3) {
    Vector v1(p2-p1);
    Vector v2(p2-p3);
    n=Cross(v2,v1);
    n.normalize();
    d=-Dot(p1, n);
}

int
Plane::Intersect( Point s, Vector v, Point& hit )
{
  double tmp;
  Point origin( 0., 0., 0. );
  Point ptOnPlane = origin - n * d;

  tmp = Dot( n, v );

  if( tmp > -1.e-6 && tmp < 1.e-6 ) // Vector v is parallel to plane
    {
      // vector from origin of line to point on plane
      
      Vector temp = s - ptOnPlane;

      if ( temp.length() < 1.e-5 )
	{
	  // the origin of plane and the origin of line are almost
	  // the same
	  
	  hit = ptOnPlane;
	  return 1;
	}
      else
	{
	  // point s and d are not the same, but maybe s lies
	  // in the plane anyways
	  
	  tmp = Dot( temp, n );

	  if(tmp > -1.e-6 && tmp < 1.e-6)
	    {
	      hit = s;
	      return 1;
	    }
	  else
	    return 0;
	}
    }

  tmp = - ( ( d + Dot( s, n ) ) / Dot( v, n ) );

#if 0  
  // the starting point s virtually lies in the plane already
  
  if ( tmp > -1.e-6 && tmp < 1.e-6 )
    hit = s;
  else
    hit = s + v * tmp;
#endif

  hit = s + v * tmp;

  return 1;
}

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.3.2.3  2000/10/26 17:55:49  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.4  2000/08/04 19:09:25  dmw
// fixed shear
//
// Revision 1.3  1999/09/08 02:26:52  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/17 06:39:27  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:56  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//
