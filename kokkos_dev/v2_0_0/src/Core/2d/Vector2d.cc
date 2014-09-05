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
 *  Vector2d.cc: ?
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
#include <Core/2d/Point2d.h>
#include <Core/2d/Vector2d.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <stdio.h>

using std::istream;
using std::ostream;

namespace SCIRun {

string
Vector2d::get_string() const
{
    char buf[100];
    sprintf(buf, "[%g, %g]", x_, y_);
    return buf;
}


Vector2d
Vector2d::normal() const
{
   Vector2d v(*this);
   v.normalize();
   return v;			// 
}

ostream& operator<<( ostream& os, const Vector2d& v )
{
   os << v.get_string();
   return os;
}

istream& operator>>( istream& is, Vector2d& v)
{
  double x, y;
  char st;
  is >> st >> x >> st >> y >> st;
  v=Vector2d(x,y);
  return is;
}

int
Vector2d::operator== ( const Vector2d& v ) const
{
    return v.x_ == x_ && v.y_ == y_;
}

int Vector2d::operator!=(const Vector2d& v) const
{
    return v.x_ != x_ || v.y_ != y_;
}

void
Pio(Piostream& stream, Vector2d& p)
{

    stream.begin_cheap_delim();
    Pio(stream, p.x_);
    Pio(stream, p.y_);
    stream.end_cheap_delim();
}


const string& 
Vector2d::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

const TypeDescription* get_type_description(Vector2d*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("Vector2d", Vector2d::get_h_file_path(), "SCIRun");
  }
  return td;
}

} // End namespace SCIRun


