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

#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class Plane {
   Vector n;
   double d;
public:
   Plane();
    Plane(const Plane &copy);
    Plane(const Point &p1, const Point &p2, const Point &p3);
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

#endif
