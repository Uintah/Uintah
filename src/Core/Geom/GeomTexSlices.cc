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
 *  GeomTexSlices.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Core/Geom/GeomTexSlices.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#ifdef _WIN32
#include <string.h>
#else
#include <strings.h>
#endif
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomTexSlices()
{
    return scinew GeomTexSlices(0,0,0,Point(0,0,0), Point(1,1,1));
}

PersistentTypeID GeomTexSlices::type_id("GeomTexSlices", "GeomObj", make_GeomTexSlices);

GeomTexSlices::GeomTexSlices(int nx, int ny, int nz, const Point& min,
			     const Point &max)
  : min(min), max(max), nx(nx), ny(ny), nz(nz), have_drawn(0), accum(0.1),
    bright(0.6)
{
    Xmajor.resize(nx, ny, nz);
    Ymajor.resize(ny, nx, nz);
    Zmajor.resize(nz, nx, ny);
}

GeomTexSlices::GeomTexSlices(const GeomTexSlices& copy)
  : GeomObj(copy)
{
}


GeomTexSlices::~GeomTexSlices()
{

}

void GeomTexSlices::get_bounds(BBox& bb)
{
  bb.extend(min);
  bb.extend(max);
}

GeomObj* GeomTexSlices::clone()
{
    return scinew GeomTexSlices(*this);
}

#define GeomTexSlices_VERSION 1

void GeomTexSlices::io(Piostream& stream)
{
    stream.begin_class("GeomTexSlices", GeomTexSlices_VERSION);
    GeomObj::io(stream);
    stream.end_class();
}    

} // End namespace SCIRun

