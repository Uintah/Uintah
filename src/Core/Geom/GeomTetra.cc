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
 *  GeomTetra.cc:  A tetrahedra object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomTetra.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomTetra()
{
    return scinew GeomTetra(Point(0,0,0), Point(0,0,1), Point(0,1,0), Point(1,0,0));
}

PersistentTypeID GeomTetra::type_id("GeomTetra", "GeomObj", make_GeomTetra);

GeomTetra::GeomTetra(const Point& p1, const Point& p2,
		     const Point& p3, const Point& p4)
: GeomObj(), p1(p1), p2(p2), p3(p3), p4(p4)
{
}

GeomTetra::GeomTetra(const GeomTetra& copy)
: GeomObj(copy), p1(copy.p1), p2(copy.p2), p3(copy.p3), p4(copy.p4)
{
}

GeomTetra::~GeomTetra()
{
}

GeomObj* GeomTetra::clone()
{
    return scinew GeomTetra(*this);
}

void GeomTetra::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
    bb.extend(p4);
}

#define GEOMTETRA_VERSION 1

void GeomTetra::io(Piostream& stream)
{

    stream.begin_class("GeomTetra", GEOMTETRA_VERSION);
    GeomObj::io(stream);
    Pio(stream, p1);
    Pio(stream, p2);
    Pio(stream, p3);
    Pio(stream, p4);
    stream.end_class();
}

} // End namespace SCIRun

