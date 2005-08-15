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
 *  LockedPolyline.cc: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Core/2d/LockedPolyline.h>
#include <Core/2d/BBox2d.h>

#include <stdio.h>

namespace SCIRun {

Persistent* make_LockedPolyline()
{
  return scinew LockedPolyline;
}

PersistentTypeID LockedPolyline::type_id("LockedPolyline", 
					 "DrawObj", 
					 make_LockedPolyline);


LockedPolyline::LockedPolyline( const vector<double> &data, const string &name)
  : Polyline( data, name )
{
}


LockedPolyline::LockedPolyline( int i )
  : Polyline(i)
{
}
  
LockedPolyline::~LockedPolyline()
{
}

void
LockedPolyline::add( double v )
{
  write_lock();
  Polyline::add(v);
  write_unlock();
}

double
LockedPolyline::at( double v )
{
  read_lock();
  double r = Polyline::at(v);
  read_unlock();

  return r;
}

#define LockedPolyline_VERSION 1

void 
LockedPolyline::io(Piostream& stream)
{
  stream.begin_class("LockedPolyline", LockedPolyline_VERSION);
  Polyline::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
