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

#include <SCICore/Geom/GeomDL.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCICore {
namespace GeomSpace {

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
    using SCICore::PersistentSpace::Pio;

    /*int version=*/ stream.begin_class("GeomDL", GEOMDL_VERSION);
    SCICore::GeomSpace::Pio(stream, child);
    stream.end_class();
}

bool GeomDL::saveobj(ostream& out, const clString& format,
			    GeomSave* saveinfo)
{
  if ( child )
    return child->saveobj(out, format, saveinfo);
  else
    return true;
}

} // End namespace GeomSpace
} // End namespace SCICore


