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
 *  Point.cc: ?
 *
 *  Written by:
 *   Author ?
 *   Department of Computer Science
 *   University of Utah
 *   Date ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Util/TypeDescription.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <stdio.h>

using std::cerr;
using std::endl;
using std::istream;
using std::ostream;

namespace SCIRun {


Point Interpolate(const Point& p1, const Point& p2, double w)
{
    return Point(
	Interpolate(p1._x, p2._x, w),
	Interpolate(p1._y, p2._y, w),
	Interpolate(p1._z, p2._z, w));
}

string Point::get_string() const
{
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", _x, _y, _z);
    return buf;
}

int Point::operator==(const Point& p) const
{
    return p._x == _x && p._y == _y && p._z == _z;
}

int Point::operator!=(const Point& p) const
{
    return p._x != _x || p._y != _y || p._z != _z;
}

Point::Point(double x, double y, double z, double w)
{
    if(w==0){
	cerr << "degenerate point!" << endl;
	_x=_y=_z=0;
    } else {
	_x=x/w;
	_y=y/w;
	_z=z/w;
    }
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2)
{
    return Point(p1._x*w1+p2._x*w2,
		 p1._y*w1+p2._y*w2,
		 p1._z*w1+p2._z*w2);
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2,
			const Point& p3, double w3)
{
    return Point(p1._x*w1+p2._x*w2+p3._x*w3,
		 p1._y*w1+p2._y*w2+p3._y*w3,
		 p1._z*w1+p2._z*w2+p3._z*w3);
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2,
			const Point& p3, double w3,
			const Point& p4, double w4)
{
    return Point(p1._x*w1+p2._x*w2+p3._x*w3+p4._x*w4,
		 p1._y*w1+p2._y*w2+p3._y*w3+p4._y*w4,
		 p1._z*w1+p2._z*w2+p3._z*w3+p4._z*w4);
}

ostream& operator<<( ostream& os, const Point& p )
{
  os << '[' << p.x() << ' ' << p.y() << ' ' << p.z() << ']';
  return os;
}

istream& operator>>( istream& is, Point& v)
{
    double x, y, z;
    char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=Point(x,y,z);
  return is;
}

int
Point::Overlap( double a, double b, double e )
{
  double hi, lo, h, l;
  
  hi = a + e;
  lo = a - e;
  h  = b + e;
  l  = b - e;

  if ( ( hi > l ) && ( lo < h ) )
    return 1;
  else
    return 0;
}
  
int
Point::InInterval( Point a, double epsilon )
{
  if ( Overlap( _x, a.x(), epsilon ) &&
      Overlap( _y, a.y(), epsilon )  &&
      Overlap( _z, a.z(), epsilon ) )
    return 1;
  else
    return 0;
}

void Pio(Piostream& stream, Point& p)
{

    stream.begin_cheap_delim();
    Pio(stream, p._x);
    Pio(stream, p._y);
    Pio(stream, p._z);
    stream.end_cheap_delim();
}

const string& 
Point::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

const TypeDescription* get_type_description(Point*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("Point", Point::get_h_file_path(), 
				"SCIRun");
  }
  return td;
}

} // End namespace SCIRun




