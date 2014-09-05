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
 * RenderMode.cc: RenderMode objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/RenderMode.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/String.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

Persistent* make_GeomRenderMode()
{
    return scinew GeomRenderMode(GeomRenderMode::WireFrame, 0);
}

PersistentTypeID GeomRenderMode::type_id("GeomRenderMode", "GeomObj", make_GeomRenderMode);

GeomRenderMode::GeomRenderMode(DrawType drawtype, GeomObj* child)
: GeomContainer(child), drawtype(drawtype)
{
}

GeomRenderMode::GeomRenderMode(const GeomRenderMode& copy)
: GeomContainer(copy), drawtype(copy.drawtype)
{
}

GeomRenderMode::~GeomRenderMode()
{
    if(child)
	delete child;
}

GeomObj* GeomRenderMode::clone()
{
    return scinew GeomRenderMode(*this);
}

void GeomRenderMode::make_prims(Array1<GeomObj*>& free,
				Array1<GeomObj*>& dontfree)
{
    if(child)
	child->make_prims(free, dontfree);
}

void GeomRenderMode::intersect(const Ray&, Material*,
			       Hit&)
{
    NOT_FINISHED("GeomRenderMode::intersect");
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

bool GeomRenderMode::saveobj(ostream&, const string&, GeomSave*)
{
    NOT_FINISHED("GeomRenderMode::saveobj");
    return false;
}

} // End namespace SCIRun

