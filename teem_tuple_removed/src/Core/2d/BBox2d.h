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
 *  BBox2d.h: 
 *
 *  Written by:
 *   Author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef BBox2d_h
#define BBox2d_h 


#include <Core/2d/Point2d.h>

namespace SCIRun {

class Piostream;

class BBox2d {

protected:
  friend void Pio( Piostream &, BBox2d& );
  
  bool have_some;
  Point2d min_, max_;

public:
  BBox2d();
  ~BBox2d();
  BBox2d(const BBox2d&);
  BBox2d(const Point2d& min, const Point2d& max);
  inline int valid() const {return have_some;}
  void reset();
  void extend(const Point2d& p);
  void extend(const BBox2d& b);

  Point2d min() const;
  Point2d max() const;

  int inside(const Point2d &p) const {
    return have_some 
      && p.x()>=min_.x() && p.y()>=min_.y() 
      && p.x()<=max_.x() && p.y()<=max_.y();
  }

};

} // End namespace SCIRun


#endif
