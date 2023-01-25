/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  Plane.h: Directed plane
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 */

#ifndef SCI_project_PolyPlane_h
#define SCI_project_PolyPlane_h 1

#include <Core/Geometry/Vector.h>

namespace Uintah {
  class Point;

class PolyPlane {
   Vector n;
   double d;
public:
    static const double NEAR_ZERO;
    PolyPlane(const PolyPlane &copy);
    PolyPlane(const Point &p1, const Point &p2, const Point &p3);
    PolyPlane(const Point &p, const Vector &n);
    PolyPlane();
    PolyPlane(double a, double b, double c, double d);
    ~PolyPlane();
    double eval_point(const Point &p) const;
    bool pointInterior(const Point & In) const;
    void flip();
    Point project(const Point& p) const;
    Vector project(const Vector& v) const;
    Vector normal() const;
    void get(double (&abcd)[4]) const;

   // changes the plane ( n and d )
   
   void ChangePlane( const Point &p1, const Point &p2, const Point &p3 );
   void ChangePlane( const Point &p1, const Vector &v); 

   // returns true if the line  v*t+s  for -inf < t < inf intersects
   // the plane.  if so, hit contains the point of intersection.

   int Intersect( Point s, Vector v, Point& hit );
   int Intersect( Point s, Vector v, double &t ) const;

   // Updated intersection routines for convex polyhedron.
   //   1/10/19 -- Justin B Hooper
   std::tuple<bool,Vector, Point> intersectWithPlane(const PolyPlane& other1) const;
   std::tuple<bool,Point> intersectWithTwoPlanes(const PolyPlane& other1, const PolyPlane& other2) const;
   std::tuple<bool,Point> intersectWithLine(const Vector& vecIn, const Point & pntIn) const;
   std::tuple<Vector,double> getDirectionAndDistance() const;
   std::tuple<Vector,Point> getDirectionAndOffset() const;
   bool isInside(const Point & pointIn) const;
};

} // End namespace Uintah

#endif
