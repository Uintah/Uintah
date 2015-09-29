/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
 *  Point.cc: ?
 *
 *  Written by:
 *   Author ?
 *   Department of Computer Science
 *   University of Utah
 *   Date ?
 *
 */

#include <Core/Util/TypeDescription.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <cstdio>

using namespace std;

namespace SCIRun {


Point Interpolate(const Point& p1, const Point& p2, double w)
{
    return Point(
	Interpolate(p1.x_, p2.x_, w),
	Interpolate(p1.y_, p2.y_, w),
	Interpolate(p1.z_, p2.z_, w));
}

string Point::get_string() const
{
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", x_, y_, z_);
    return buf;
}

int Point::operator==(const Point& p) const
{
    return p.x_ == x_ && p.y_ == y_ && p.z_ == z_;
}

int Point::operator!=(const Point& p) const
{
    return p.x_ != x_ || p.y_ != y_ || p.z_ != z_;
}

Point::Point(double x, double y, double z, double w)
{
    if(w==0){
	cerr << "degenerate point!" << endl;
	x_=y_=z_=0;
    } else {
	x_=x/w;
	y_=y/w;
	z_=z/w;
    }
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2)
{
    return Point(p1.x_*w1+p2.x_*w2,
		 p1.y_*w1+p2.y_*w2,
		 p1.z_*w1+p2.z_*w2);
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2,
			const Point& p3, double w3)
{
    return Point(p1.x_*w1+p2.x_*w2+p3.x_*w3,
		 p1.y_*w1+p2.y_*w2+p3.y_*w3,
		 p1.z_*w1+p2.z_*w2+p3.z_*w3);
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2,
			const Point& p3, double w3,
			const Point& p4, double w4)
{
    return Point(p1.x_*w1+p2.x_*w2+p3.x_*w3+p4.x_*w4,
		 p1.y_*w1+p2.y_*w2+p3.y_*w3+p4.y_*w4,
		 p1.z_*w1+p2.z_*w2+p3.z_*w3+p4.z_*w4);
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
  if ( Overlap( x_, a.x(), epsilon ) &&
      Overlap( y_, a.y(), epsilon )  &&
      Overlap( z_, a.z(), epsilon ) )
    return 1;
  else
    return 0;
}

void Pio(Piostream& stream, Point& p)
{

    stream.begin_cheap_delim();
    Pio(stream, p.x_);
    Pio(stream, p.y_);
    Pio(stream, p.z_);
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
				"SCIRun", TypeDescription::DATA_E);
  }
  return td;
}

} // End namespace SCIRun
