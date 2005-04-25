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


