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

