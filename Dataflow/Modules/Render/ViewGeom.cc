/*
 *  ViewGeom.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <iostream>
using std::cerr;
using std::ostream;

#include <Core/Geom/GeomOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/BBoxCache.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Modules/Render/ViewGeom.h>
#include <Dataflow/Modules/Render/ViewWindow.h>

namespace SCIRun {


Persistent* make_GeomViewerItem()
{
    return scinew GeomViewerItem;
}

PersistentTypeID GeomViewerItem::type_id("GeomViewerItem", "GeomObj",
					 make_GeomViewerItem);

GeomViewerPort::GeomViewerPort(int no)
:portno(no),msg_head(0),msg_tail(0)
{
    // just use default constructor for base class...
}

GeomViewerPort::~GeomViewerPort()
{
    // maybee flush mesages, or do nothing...
}

GeomViewerItem::GeomViewerItem()
:child(0),lock(0)
{
}

GeomViewerItem::GeomViewerItem(GeomObj* obj,const clString& nm, 
			       CrowdMonitor* lck)
:child(obj),name(nm),lock(lck)
{
    if (!lock)
	child = new GeomBBoxCache(obj);
}

GeomViewerItem::~GeomViewerItem()
{
    if (child)
	delete child;  // I'm not sure if this should be here...
}

void GeomViewerItem::get_triangles( Array1<float> &v)
{
  if ( child )
    child->get_triangles(v);
}

GeomObj* GeomViewerItem::clone()
{
    cerr << "GeomViewerItem::clone not implemented!\n";
    return 0;
}

void GeomViewerItem::reset_bbox()
{
    child->reset_bbox();
}

void GeomViewerItem::get_bounds(BBox& box)
{
    child->get_bounds(box);
}

#define GeomViewerITEM_VERSION 1

void GeomViewerItem::io(Piostream& stream)
{

    stream.begin_class("GeomViewerItem", GeomViewerITEM_VERSION);
    int have_lock;
    if(stream.writing())
	have_lock=lock?1:0;
    Pio(stream, have_lock);
    if(stream.reading())
	if(have_lock)
	    lock=new CrowdMonitor("GeomViewerItem crowd monitor");
	else
	    lock=0;
    Pio(stream, name);
    Pio(stream, child);
    stream.end_class();
}

bool GeomViewerItem::saveobj(ostream& out, const clString& format,
			     GeomSave* saveinfo)
{
    return child->saveobj(out, format, saveinfo);
}

} // End namespace SCIRun
