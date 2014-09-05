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
 *  Vector.cc: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Util/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_off.h>
#include <stdio.h>

using std::istream;
using std::ostream;

namespace SCIRun {

string
Vector::get_string() const
{
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", _x, _y, _z);
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
    return v._x == _x && v._y == _y && v._z == _z;
}

int Vector::operator!=(const Vector& v) const
{
    return v._x != _x || v._y != _y || v._z != _z;
}

void
Pio(Piostream& stream, Vector& p)
{

    stream.begin_cheap_delim();
    Pio(stream, p._x);
    Pio(stream, p._y);
    Pio(stream, p._z);
    stream.end_cheap_delim();
}

void Vector::rotz90(const int c)
{
    // Rotate by c*90 degrees counter clockwise
    switch(c%4){
    case 0:
	// 0 degrees, do nothing
	break;
    case 1:
	// 90 degrees
	{
	    double newx=-_y;
	    _y=_x;
	    _x=newx;
	}
	break;
    case 2:
	// 180 degrees
	_x=-_x;
	_y=-_y;
	break;
    case 3:
	// 270 degrees
	{
	    double newy=-_x;
	    _x=_y;
	    _y=newy;
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
    td = scinew TypeDescription("Vector", Vector::get_h_file_path(), "SCIRun");
  }
  return td;
}

} // End namespace SCIRun


