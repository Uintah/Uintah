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

  
