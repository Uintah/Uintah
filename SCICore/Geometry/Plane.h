
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

#include <share/share.h>

#include <Geometry/Point.h>
#include <Geometry/Vector.h>

namespace SCICore {
namespace Geometry {

class SHARE Plane {
   Vector n;
   double d;
public:
    Plane(const Plane &copy);
    Plane(const Point &p1, const Point &p2, const Point &p3);
    Plane();
    ~Plane();
    double eval_point(const Point &p);
    void flip();
    Point project(const Point& p);
   Vector normal();

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
