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
 *  Sticky.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include "Sticky.h"
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent *make_GeomSticky() {
  return scinew GeomSticky( 0 );
}

PersistentTypeID GeomSticky::type_id("GeomSticky", "GeomObj", make_GeomSticky);

GeomSticky::GeomSticky( GeomObj *c )
  : GeomObj(), child(c)
{
}

GeomSticky::GeomSticky( const GeomSticky &copy )
  : GeomObj(copy), child(copy.child)
{
}

GeomSticky::~GeomSticky()
{
  if(child)
    delete child;
}

GeomObj* GeomSticky::clone() {
  return scinew GeomSticky( *this );
}

void GeomSticky::get_bounds( BBox& bb ) {
  child->get_bounds( bb );
}

#define GeomSticky_VERSION 1

void GeomSticky::io(Piostream& stream) {
  stream.begin_class("GeomSticky", GeomSticky_VERSION);
  GeomObj::io(stream);
  Pio(stream, child);
  stream.end_class();
}

bool GeomSticky::saveobj(ostream&, const string&, GeomSave*) {
  NOT_FINISHED("GeomSticky::saveobj");
  return false;
}

} // End namespace SCIRun

