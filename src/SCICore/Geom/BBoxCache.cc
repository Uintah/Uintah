//static char *id="@(#) $Id$"

/*
 *  BBoxCache.cc: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Geom/BBoxCache.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomBBoxCache()
{
    return scinew GeomBBoxCache(0);
}

PersistentTypeID GeomBBoxCache::type_id("GeomBBoxCache", "GeomObj",
					make_GeomBBoxCache);


GeomBBoxCache::GeomBBoxCache(GeomObj* obj)
:child(obj),bbox_cached(0)
{

}

GeomBBoxCache::GeomBBoxCache(GeomObj* obj, const BBox &box)
:child(obj),bbox_cached(1)
{
  bbox.extend( box );
}

GeomBBoxCache::~GeomBBoxCache()
{
    if(child)
	delete child;
}

void GeomBBoxCache::get_triangles( Array1<float> &v)
{
    if ( child )
      child->get_triangles(v);
}

GeomObj* GeomBBoxCache::clone()
{
    cerr << "GeomBBoxCache::clone not implemented!\n";
    return 0;
}

void GeomBBoxCache::reset_bbox()
{
    bbox_cached = 0;
}

void GeomBBoxCache::get_bounds(BBox& box)
{
  if (!bbox_cached || !bbox.valid()) {
    bbox.reset();
    child->get_bounds(bbox);
    bbox_cached = 1;
  }
  
  box.extend( bbox );
}

#define GEOMBBOXCACHE_VERSION 2

void GeomBBoxCache::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;

    int version=stream.begin_class("GeomBBoxCache", GEOMBBOXCACHE_VERSION);
    Pio(stream, bbox_cached);
    if(version < 2){
	int bsphere_cached;
	Pio(stream, bsphere_cached);
    }
    SCICore::Geometry::Pio(stream, bbox);
    if(version < 2){
	// Old BSphere stuff...
	stream.begin_cheap_delim();
	int have_some;
	Pio(stream, have_some);
	Point cen;
	SCICore::Geometry::Pio(stream, cen);
	double rad;
	Pio(stream, rad);
	stream.end_cheap_delim();
    }
    SCICore::GeomSpace::Pio(stream, child);
    stream.end_class();
}

bool GeomBBoxCache::saveobj(ostream& out, const clString& format,
			    GeomSave* saveinfo)
{
    return child->saveobj(out, format, saveinfo);
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.7.2.3  2000/10/26 17:18:35  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.9  2000/07/06 23:18:55  yarden
// fix a bug if the bbox is not valid
//
// Revision 1.8  2000/06/06 16:01:42  dahart
// - Added get_triangles() to several classes for serializing triangles to
// send them over a network connection.  This is a short term (hack)
// solution meant for now to allow network transport of the geometry that
// Yarden's modules produce.  Yarden has promised to work on a more
// general solution to network serialization of SCIRun geometry objects. ;)
//
// Revision 1.7  1999/10/07 02:07:39  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/08/29 00:46:53  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/28 17:54:38  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/19 23:18:05  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/17 23:50:17  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:03  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:36  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:48  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

