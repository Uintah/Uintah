//static char *id="@(#) $Id$";

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
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::PersistentSpace::Pio;

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

void GeomSticky::get_bounds(BSphere& sp) {
  child->get_bounds( sp );
}

void GeomSticky::make_prims( Array1<GeomObj*>&, Array1<GeomObj*>&)
{
  NOT_FINISHED("GeomSticky::make_prims");
}

void GeomSticky::preprocess() {
  NOT_FINISHED("GeomSticky::preprocess");
}

void GeomSticky::intersect(const Ray&, Material*, Hit&) {
  NOT_FINISHED("GeomSticky::intersect");
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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:22  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:51  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//
