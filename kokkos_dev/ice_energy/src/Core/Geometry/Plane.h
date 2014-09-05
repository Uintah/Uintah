/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Plane_h
#define SCI_project_Plane_h 1

#include <Core/Geometry/Vector.h>

namespace SCIRun {
  class Point;
class Plane {
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

} // End namespace SCIRun

#endif
