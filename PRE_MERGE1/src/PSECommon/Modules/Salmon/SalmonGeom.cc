//static char *id="@(#) $Id$";

/*
 *  SalmonGeom.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Geom/GeomOpenGL.h>
#include <Malloc/Allocator.h>
#include <Modules/Salmon/SalmonGeom.h>
#include <Geom/BBoxCache.h>
#include <iostream.h>
#include <Modules/Salmon/Roe.h>

namespace PSECommon {
namespace Modules {

using SCICore::PersistentSpace::Persistent;
using SCICore::PersistentSpace::Persistent;
using SCICore::GeomSpace::GeomBBoxCache;

Persistent* make_GeomSalmonItem()
{
    return scinew GeomSalmonItem;
}

PersistentTypeID GeomSalmonItem::type_id("GeomSalmonItem", "GeomObj",
					 make_GeomSalmonItem);

GeomSalmonPort::GeomSalmonPort(int no)
:portno(no),msg_head(0),msg_tail(0)
{
    // just use default constructor for base class...
}

GeomSalmonPort::~GeomSalmonPort()
{
    // maybee flush mesages, or do nothing...
}

GeomSalmonItem::GeomSalmonItem()
:child(0),lock(0)
{
}

GeomSalmonItem::GeomSalmonItem(GeomObj* obj,const clString& nm, 
			       CrowdMonitor* lck)
:child(obj),name(nm),lock(lck)
{
    if (!lock)
	child = new GeomBBoxCache(obj);
}

GeomSalmonItem::~GeomSalmonItem()
{
    if (child)
	delete child;  // I'm not sure if this should be here...
}

GeomObj* GeomSalmonItem::clone()
{
    cerr << "GeomSalmonItem::clone not implemented!\n";
    return 0;
}

void GeomSalmonItem::reset_bbox()
{
    child->reset_bbox();
}

void GeomSalmonItem::get_bounds(BBox& box)
{
    child->get_bounds(box);
}

void GeomSalmonItem::get_bounds(BSphere& sphere)
{
    child->get_bounds(sphere);
}

void GeomSalmonItem::make_prims(Array1<GeomObj *>& free ,
				Array1<GeomObj *>& dontfree )
{
    child->make_prims(free,dontfree);
}

void GeomSalmonItem::preprocess()
{
    child->preprocess();
}

void GeomSalmonItem::intersect(const Ray& ray, Material* m,
			       Hit& hit)
{
    child->intersect(ray,m,hit);
}

#define GEOMSALMONITEM_VERSION 1

void GeomSalmonItem::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("GeomSalmonItem", GEOMSALMONITEM_VERSION);
    int have_lock;
    if(stream.writing())
	have_lock=lock?1:0;
    Pio(stream, have_lock);
    if(stream.reading())
	if(have_lock)
	    lock=new CrowdMonitor;
	else
	    lock=0;
    Pio(stream, name);
    Pio(stream, child);
    stream.end_class();
}

bool GeomSalmonItem::saveobj(ostream& out, const clString& format,
			     GeomSave* saveinfo)
{
    return child->saveobj(out, format, saveinfo);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:53  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
