
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
#include <Core/Containers/String.h>
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

bool GeomSticky::saveobj(ostream&, const clString&, GeomSave*) {
  NOT_FINISHED("GeomSticky::saveobj");
  return false;
}

} // End namespace SCIRun

