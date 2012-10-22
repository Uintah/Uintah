/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
 *  Vector.cc: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 */

#include <Core/Util/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <cstdio>

using std::istream;
using std::ostream;

namespace SCIRun {

string
Vector::get_string() const
{
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", x_, y_, z_);
    return buf;
}

void
Vector::find_orthogonal(Vector& v1, Vector& v2) const
{
    Vector v0(Cross(*this, Vector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(*this, Vector(0,1,0));
    }
    v1=Cross(*this, v0);
    v1.normalize();
    v2=Cross(*this, v1);
    v2.normalize();
}

bool
Vector::check_find_orthogonal(Vector& v1, Vector& v2) const
{
    Vector v0(Cross(*this, Vector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(*this, Vector(0,1,0));
    }
    v1=Cross(*this, v0);
    double length1 = v1.length();
    if(length1 == 0)
       return false;
    v1 *= 1./length1;
    v2=Cross(*this, v1);
    double length2 = v2.length();
    if(length2 == 0)
       return false;
    v2 *= 1./length2;
    return true;
}

Vector
Vector::normal() const
{
   Vector v(*this);
   v.normalize();
   return v;			// 
}

ostream& operator<<( ostream& os, const Vector& v )
{
  os << '[' << v.x() << ' ' << v.y() << ' ' << v.z() << ']';
  return os;
}

istream& operator>>( istream& is, Vector& v)
{
  double x, y, z;
  char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=Vector(x,y,z);
  return is;
}

int
Vector::operator== ( const Vector& v ) const
{
    return v.x_ == x_ && v.y_ == y_ && v.z_ == z_;
}

int Vector::operator!=(const Vector& v) const
{
    return v.x_ != x_ || v.y_ != y_ || v.z_ != z_;
}

void
Pio(Piostream& stream, Vector& p)
{

    stream.begin_cheap_delim();
    Pio(stream, p.x_);
    Pio(stream, p.y_);
    Pio(stream, p.z_);
    stream.end_cheap_delim();
}

void
Vector::rotz90(const int c)
{
    // Rotate by c*90 degrees counter clockwise
    switch(c%4){
    case 0:
	// 0 degrees, do nothing
	break;
    case 1:
	// 90 degrees
	{
	    double newx=-y_;
	    y_=x_;
	    x_=newx;
	}
	break;
    case 2:
	// 180 degrees
	x_=-x_;
	y_=-y_;
	break;
    case 3:
	// 270 degrees
	{
	    double newy=-x_;
	    x_=y_;
	    y_=newy;
	}
	break;
    }
}

const string& 
Vector::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

const TypeDescription* get_type_description(Vector*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("Vector", Vector::get_h_file_path(), "SCIRun", 
				TypeDescription::DATA_E);
  }
  return td;
}

} // End namespace SCIRun


