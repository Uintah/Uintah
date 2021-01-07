/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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
 *  FloatPlane.h: Directed plane
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 */

#ifndef SCI_project_FloatPlane
#define SCI_project_FloatPlane 1

#include <Core/Geometry/FloatVector.h>

namespace Uintah {
  class FloatPoint;
class FloatPlane {
   FloatVector n;
   float d;
public:
    FloatPlane(const FloatPlane &copy);
    FloatPlane(const FloatPoint &p1, const FloatPoint &p2, const FloatPoint &p3);
    FloatPlane(const FloatPoint &p, const FloatVector &n);
    FloatPlane();
    FloatPlane(float a, float b, float c, float d);
    ~FloatPlane();
    float eval_point(const FloatPoint &p) const;
    void flip();
    FloatPoint project(const FloatPoint& p) const;
    FloatVector project(const FloatVector& v) const;
    FloatVector normal() const;
    void get(float (&abcd)[4]) const;

   // changes the plane ( n and d )
   
   void ChangePlane( const FloatPoint &p1, const FloatPoint &p2, const FloatPoint &p3 );
   void ChangePlane( const FloatPoint &p1, const FloatVector &v); 

   // returns true if the line  v*t+s  for -inf < t < inf intersects
   // the plane.  if so, hit contains the point of intersection.

   int Intersect( FloatPoint s, FloatVector v, FloatPoint& hit );
   int Intersect( FloatPoint s, FloatVector v, float &t ) const;
};

} // End namespace Uintah

#endif
