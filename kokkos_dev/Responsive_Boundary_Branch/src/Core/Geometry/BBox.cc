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
 *  BBox.cc: Bounding box class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   1994
 *
 *  Copyright (C) 1994,2004 SCI Group
 */

#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Assert.h>
#include <Core/Math/MinMax.h>
#include <Core/Persistent/Persistent.h>
#include <cstdlib>

using namespace SCIRun;
using namespace std;

BBox::BBox()
{
  is_valid = false;
}

BBox::BBox(const Point& min, const Point& max)
  : cmin(min), cmax(max), is_valid(true)
{
}

BBox::BBox(const BBox& copy)
: cmin(copy.cmin), cmax(copy.cmax), is_valid(copy.is_valid)
{
}

BBox::~BBox()
{
}

BBox& BBox::operator=(const BBox& copy)
{
  is_valid = copy.is_valid;
  cmin = copy.cmin;
  cmax = copy.cmax;
  return *this;
}
    
void BBox::reset()
{
  is_valid = false;
}

void BBox::extend(const Point& p)
{
  if(is_valid){
    cmin=Min(p, cmin);
    cmax=Max(p, cmax);
  } else {
    cmin=p;
    cmax=p;
    is_valid = true;
  }
}

void BBox::extend(const Point& p, double radius)
{
  Vector r(radius,radius,radius);
  if(is_valid){
    cmin=Min(p-r, cmin);
    cmax=Max(p+r, cmax);
  } else {
    cmin=p-r;
    cmax=p+r;
    is_valid = true;
  }
}

void BBox::extend(const BBox& b)
{
  if(b.valid()){
    extend(b.min());
    extend(b.max());
  }
}

void BBox::extend_disc(const Point& cen, const Vector& normal, double r)
{
  if (normal.length2() < 1.e-6) { extend(cen); return; }
  Vector n(normal.normal());
  double x=Sqrt(1-n.x())*r;
  double y=Sqrt(1-n.y())*r;
  double z=Sqrt(1-n.z())*r;
  extend(cen+Vector(x,y,z));
  extend(cen-Vector(x,y,z));
}

Point BBox::center() const
{
  ASSERT(is_valid);
  return Interpolate(cmin, cmax, 0.5);
}

void BBox::translate(const Vector &v)
{
  cmin+=v;
  cmax+=v;
}

void BBox::scale(double s, const Vector&o)
{
  cmin-=o;
  cmax-=o;
  cmin*=s;
  cmax*=s;
  cmin+=o;
  cmax+=o;
}

double BBox::longest_edge()
{
  ASSERT(is_valid);
  Vector diagonal(cmax-cmin);
  return Max(diagonal.x(), diagonal.y(), diagonal.z());
}

Point BBox::min() const
{
  ASSERT(is_valid);
  return cmin;
}

Point BBox::max() const
{
  ASSERT(is_valid);
  return cmax;
}

Vector BBox::diagonal() const
{
  ASSERT(is_valid);
  return cmax-cmin;
}

bool BBox::overlaps(const BBox & bb)
{
  if( bb.cmin.x() > cmax.x() || bb.cmax.x() < cmin.x())
    return false;
  else if( bb.cmin.y() > cmax.y() || bb.cmax.y() < cmin.y())
    return false;
  else if( bb.cmin.z() > cmax.z() || bb.cmax.z() < cmin.z())
    return false;

  return true;
}

bool BBox::overlaps2(const BBox & bb)
{
  if( bb.cmin.x() >= cmax.x() || bb.cmax.x() <= cmin.x())
    return false;
  else if( bb.cmin.y() >= cmax.y() || bb.cmax.y() <= cmin.y())
    return false;
  else if( bb.cmin.z() >= cmax.z() || bb.cmax.z() <= cmin.z())
    return false;

  return true;
}

bool BBox::intersect(const Point& origin, const Vector& dir,
                     Point& hitPoint)
{
  Vector t1 = (cmin - origin) / dir;
  Vector t2 = (cmax - origin) / dir;
  Vector tn = Min(t1, t2);
  Vector tf = Max(t1, t2);
  double tnear = tn.maxComponent();
  double tfar = tf.minComponent();
  if(tnear <= tfar){
    hitPoint = origin + dir*tnear;
    return true;
  } else {
    return false;
  }
}

namespace SCIRun {

  void Pio(Piostream & stream, BBox & box)
  {
    stream.begin_cheap_delim();
    
    // Store the valid flag as an int, because old files did and we
    // must be backward compatible
    int tmp = box.is_valid;
    Pio(stream, tmp);
    if(stream.reading())
      box.is_valid = tmp;
    if (box.is_valid) {
      Pio(stream, box.cmin);
      Pio(stream, box.cmax);
    }
    stream.end_cheap_delim();
  }


  ostream&
  operator<<(ostream& out, const BBox& b)
  {
    out << b.cmin << ".." << b.cmax;
    return out;
  }

} // End namespace SCIRun
