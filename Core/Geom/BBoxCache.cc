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
