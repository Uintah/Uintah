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
 *  Point2d.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Util/TypeDescription.h>
#include <Core/2d/Point2d.h>
#include <Core/2d/Vector2d.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <stdio.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>


using std::istream;
using std::ostream;

namespace SCIRun {


Point2d::Point2d(const Vector2d& v)
    : x_(v.x()), y_(v.y())
{
}


Point2d Interpolate(const Point2d& p1, const Point2d& p2, double w)
{
  return Point2d(
		 Interpolate(p1.x_, p2.x_, w),
		 Interpolate(p1.y_, p2.y_, w)
		 );
}

string Point2d::get_string() const
{
  char buf[100];
  sprintf(buf, "[%g, %g]", x_, y_);
  return buf;
}

int Point2d::operator==(const Point2d& p) const
{
  return p.x_ == x_ && p.y_ == y_ ;
}

int Point2d::operator!=(const Point2d& p) const
{
  return p.x_ != x_ || p.y_ != y_ ;
}

Point2d 
Point2d::operator+(const Vector2d& v) const
{
  return Point2d( x_ + v.x_ ,  y_ + v.y_ ) ;
}

Point2d 
Point2d::operator-(const Vector2d& v) const
{
  return Point2d( x_ - v.x_ ,  y_ - v.y_ ) ;
}



ostream& operator<<( ostream& os, const Point2d& p )
{
   os << p.get_string();
   return os;
}

istream& operator>>( istream& is, Point2d& v)
{
  double x, y;
  char st;
  is >> st >> x >> st >> y >> st;
  v=Point2d(x,y);
  return is;
}


void Pio(Piostream& stream, Point2d& p)
{
  stream.begin_cheap_delim();
  Pio(stream, p.x_);
  Pio(stream, p.y_);
  stream.end_cheap_delim();
}

const string& 
Point2d::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

const TypeDescription* get_type_description(Point2d*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("Point2d", Point2d::get_h_file_path(), 
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun




