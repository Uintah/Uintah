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

#include <Core/Geom/BBoxCache.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomBBoxCache()
{
    return scinew GeomBBoxCache(0);
}

PersistentTypeID GeomBBoxCache::type_id("GeomBBoxCache", "GeomObj",
					make_GeomBBoxCache);


GeomBBoxCache::GeomBBoxCache(GeomHandle obj)
 : GeomContainer(obj), bbox_cached(0)
{

}

GeomBBoxCache::GeomBBoxCache(GeomHandle obj, const BBox &box)
  : GeomContainer(obj), bbox_cached(true)
{
  bbox.extend( box );
}

GeomObj* GeomBBoxCache::clone()
{
    cerr << "GeomBBoxCache::clone not implemented!\n";
    return 0;
}

void GeomBBoxCache::reset_bbox()
{
    bbox_cached = false;
    GeomContainer::reset_bbox();
}

void GeomBBoxCache::get_bounds(BBox& box)
{
  if (!bbox_cached || !bbox.valid()) {
    bbox.reset();
    child_->get_bounds(bbox);
    bbox_cached = true;
  }
  
  box.extend( bbox );
}

#define GEOMBBOXCACHE_VERSION 2

void GeomBBoxCache::io(Piostream& stream)
{

    int version=stream.begin_class("GeomBBoxCache", GEOMBBOXCACHE_VERSION);
    Pio(stream, bbox_cached);
    if(version < 2){
	int bsphere_cached;
	Pio(stream, bsphere_cached);
    }
    Pio(stream, bbox);
    if(version < 2){
	// Old BSphere stuff...
	stream.begin_cheap_delim();
	int have_some;
	Pio(stream, have_some);
	Point cen;
	Pio(stream, cen);
	double rad;
	Pio(stream, rad);
	stream.end_cheap_delim();
    }
    Pio(stream, child_);
    stream.end_class();
}

} // End namespace SCIRun
