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
 *  GeomDL.cc: Create a display list for its child
 *
 *  Written by:
 *   Author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Date July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Geom/GeomDL.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomDL()
{
    return scinew GeomDL(0);
}

PersistentTypeID GeomDL::type_id("GeomDL", "GeomObj",
					make_GeomDL);


GeomDL::GeomDL(GeomObj* obj)
  :child(obj), have_dl(false)
{
}


GeomDL::~GeomDL()
{
  if(child)
    delete child;
}

void GeomDL::get_triangles( Array1<float> &v)
{
    if ( child )
      child->get_triangles(v);
}

GeomObj* GeomDL::clone()
{
    cerr << "GeomDL::clone not implemented!\n";
    return 0;
}


void GeomDL::get_bounds(BBox& box)
{
  if ( child )
    child->get_bounds(box);
}

#define GEOMDL_VERSION 1

void GeomDL::io(Piostream& stream)
{

    /*int version=*/ stream.begin_class("GeomDL", GEOMDL_VERSION);
    Pio(stream, child);
    stream.end_class();
}

bool GeomDL::saveobj(ostream& out, const string& format,
			    GeomSave* saveinfo)
{
  if ( child )
    return child->saveobj(out, format, saveinfo);
  else
    return true;
}

} // End namespace SCIRun


