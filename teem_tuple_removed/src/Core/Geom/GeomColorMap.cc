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
 *  GeomColorMap.cc: Set colormap for indexed color primitives.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Core/Geom/GeomColorMap.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomColorMap()
{
  return scinew GeomColorMap(0, 0);
}

PersistentTypeID GeomColorMap::type_id("GeomColorMap", "GeomContainer",
				       make_GeomColorMap);


GeomColorMap::GeomColorMap(GeomHandle obj, ColorMapHandle cmap)
  : GeomContainer(obj), cmap_(cmap)
{
}

GeomColorMap::GeomColorMap(const GeomColorMap &copy)
  : GeomContainer(copy), cmap_(copy.cmap_)
{
}


GeomObj*
GeomColorMap::clone()
{
  return scinew GeomColorMap(*this);
}


#define GEOMCOLORMAP_VERSION 1

void
GeomColorMap::io(Piostream& stream)
{
  stream.begin_class("GeomColorMap", GEOMCOLORMAP_VERSION);
  Pio(stream, cmap_);
  Pio(stream, child_);
  stream.end_class();
}

} // End namespace SCIRun


