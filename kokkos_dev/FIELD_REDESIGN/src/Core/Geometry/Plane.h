
/*
 *  Plane.h: Directed plane
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Plane_h
#define SCI_project_Plane_h 1

#include <SCICore/share/share.h>

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

namespace SCICore {
namespace Geometry {

class SCICORESHARE Plane {
   Vector n;
   double d;
public:
    Plane(const Plane &copy);
    Plane(const Point &p1, const Point &p2, const Point &p3);
    Plane();
    Plane(double a, double b, double c, double d);
    ~Plane();
    double eval_point(const Point &p) const;
    void flip();
    Point project(const Point& p) const;
    Vector project(const Vector& v) const;
    Vector normal() const;

   // changes the plane ( n and d )
   
   void ChangePlane( const Point &p1, const Point &p2, const Point &p3 );

   // returns true if the line  v*t+s  for -inf < t < inf intersects
   // the plane.  if so, hit contains the point of intersection.

   int Intersect( Point s, Vector v, Point& hit );
};

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.2.2.2  2000/10/26 17:55:49  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.3  2000/08/04 19:09:25  dmw
// fixed shear
//
// Revision 1.2  1999/08/17 06:39:28  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:56  mcq
// Initial commit
//
// Revision 1.4  1999/07/09 00:27:39  moulding
// added SHARE support for win32 shared libraries (.dll's)
//
// Revision 1.3  1999/05/06 19:56:16  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:18  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif
