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
 * GeomRenderMode.cc: RenderMode objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomRenderMode.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomRenderMode()
{
    return scinew GeomRenderMode(GeomRenderMode::WireFrame, 0);
}

PersistentTypeID GeomRenderMode::type_id("GeomRenderMode", "GeomObj", make_GeomRenderMode);

GeomRenderMode::GeomRenderMode(DrawType drawtype, GeomHandle child)
: GeomContainer(child), drawtype(drawtype)
{
}

GeomRenderMode::GeomRenderMode(const GeomRenderMode& copy)
: GeomContainer(copy), drawtype(copy.drawtype)
{
}

GeomObj* GeomRenderMode::clone()
{
    return scinew GeomRenderMode(*this);
}

#define GEOMRENDERMODE_VERSION 1

void GeomRenderMode::io(Piostream& stream)
{
    stream.begin_class("GeomRenderMode", GEOMRENDERMODE_VERSION);
    GeomContainer::io(stream);
    int tmp=drawtype;
    Pio(stream, tmp);
    if(stream.reading())
	drawtype=(DrawType)tmp;
    stream.end_class();
}

} // End namespace SCIRun


