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
 *  BBox.h: ?
 *
 *  Written by:
 *   Author ?
 *   Department of Computer Science
 *   University of Utah
 *   Date ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_BBox_h
#define Geometry_BBox_h 1

#include <Core/share/share.h>

#include <Core/Geometry/Point.h>

namespace SCIRun {

#define EEpsilon  1e-13

class Vector;
class Piostream;

class SCICORESHARE BBox {

protected:
  friend SCICORESHARE void Pio( Piostream &, BBox& );

  int have_some;
  Point cmin;
  Point cmax;
  Point bcmin, bcmax;
  Point extracmin;
  int inbx, inby, inbz;

public:
  BBox();
  ~BBox();
  BBox(const BBox&);
  BBox(const Point& min, const Point& max);
  inline int valid() const {return have_some;}
  void reset();
  void extend(const Point& p);
  void extend(const Point& p, double radius);
  void extend(const BBox& b);
  void extend_disc(const Point& cen, const Vector& normal, double r);
  Point center() const;
  double longest_edge();
  void translate(const Vector &v);
  void scale(double s, const Vector &o);
  Point min() const;
  Point max() const;
  Vector diagonal() const;

  inline int inside(const Point &p) const {return (have_some && p.x()>=cmin.x() && p.y()>=cmin.y() && p.z()>=cmin.z() && p.x()<=cmax.x() && p.y()<=cmax.y() && p.z()<=cmax.z());}

  // prepares for intersection by assigning the closest bbox corner
  // to extracmin and initializing an epsilon bigger bbox
    
  void PrepareIntersect( const Point& e );
    
  // returns true if the ray hit the bbox and returns the hit point
  // in hitNear

  int Intersect( const Point& e, const Vector& v, Point& hitNear );

  // given a t distance, assigns the hit point.  returns true
  // if the hit point lies on a bbox face

  int TestTx( const Point& e, const Vector& v, double tx, Point& hitNear );
  int TestTy( const Point& e, const Vector& v, double ty, Point& hitNear );
  int TestTz( const Point& e, const Vector& v, double tz, Point& hitNear );

  bool Overlaps( const BBox& bb );
};

} // End namespace SCIRun


#endif
