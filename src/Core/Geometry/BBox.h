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
