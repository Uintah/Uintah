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
 *  BBox2d.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/2d/BBox2d.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {

BBox2d::BBox2d()
{
    have_some=0;
}

BBox2d::BBox2d(const Point2d& min, const Point2d& max):
  have_some(true), min_(min), max_(max)
{
}

BBox2d::BBox2d(const BBox2d& copy)
: have_some(copy.have_some), min_(copy.min_), max_(copy.max_) 
{
}
    
BBox2d::~BBox2d()
{
}

void BBox2d::reset()
{
  have_some=false;
}

void BBox2d::extend(const Point2d& p)
{
  if(have_some){
    min_=Min(p, min_);
    max_=Max(p, max_);
  } else {
    min_=p;
    max_=p;
    have_some=true;
    }
}

void BBox2d::extend(const BBox2d& b)
{
  if(b.valid()){
    extend(b.min());
    extend(b.max());
  }
}


Point2d BBox2d::min() const
{
    ASSERT(have_some);
    return min_;
}

Point2d BBox2d::max() const
{
  ASSERT(have_some);
  return max_;
}

void Pio(Piostream & stream, BBox2d & box) 
{
  stream.begin_cheap_delim();
  Pio(stream, box.have_some);
  if (box.have_some) {
    Pio(stream, box.min_);
    Pio(stream, box.max_);
  }
  stream.end_cheap_delim();
}

} // End namespace SCIRun

