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
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream.h>

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomBBoxCache()
{
    return scinew GeomBBoxCache(0);
}

PersistentTypeID GeomBBoxCache::type_id("GeomBBoxCache", "GeomObj",
					make_GeomBBoxCache);


GeomBBoxCache::GeomBBoxCache(GeomObj* obj)
:child(obj),bbox_cached(0),bsphere_cached(0)
{

}

GeomBBoxCache::GeomBBoxCache(GeomObj* obj, BBox &box)
:child(obj),bbox_cached(1),bsphere_cached(0)
{
  bbox.extend( box );
}

GeomBBoxCache::~GeomBBoxCache()
{
    if(child)
	delete child;
}

GeomObj* GeomBBoxCache::clone()
{
    cerr << "GeomBBoxCache::clone not implemented!\n";
    return 0;
}

void GeomBBoxCache::reset_bbox()
{
    bbox_cached = bsphere_cached = 0;
}

void GeomBBoxCache::get_bounds(BBox& box)
{
    if (!bbox_cached) {
	bbox.reset();
	child->get_bounds(bbox);
	bbox_cached = 1;
    }

    box.extend( bbox );
}

void GeomBBoxCache::get_bounds(BSphere& sphere)
{
    if (!bsphere_cached) {
	child->get_bounds(sphere);
	bsphere = sphere;
	bsphere_cached = 1;
    }
    else {
	sphere = bsphere;
    }
}


void GeomBBoxCache::make_prims(Array1<GeomObj *>& free ,
				Array1<GeomObj *>& dontfree )
{
    child->make_prims(free,dontfree);
}

void GeomBBoxCache::preprocess()
{
    child->preprocess();
}

void GeomBBoxCache::intersect(const Ray& ray, Material* m,
			       Hit& hit)
{
    child->intersect(ray,m,hit);
}

#define GEOMBBOXCACHE_VERSION 1

void GeomBBoxCache::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
    using SCICore::GeomSpace::Pio;

    stream.begin_class("GeomBBoxCache", GEOMBBOXCACHE_VERSION);
    Pio(stream, bbox_cached);
    Pio(stream, bsphere_cached);
    Pio(stream, bbox);
    Pio(stream, bsphere);
    Pio(stream, child);
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

