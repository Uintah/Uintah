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
  : msg_head(0), msg_tail(0), portno(no)
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

GeomViewerItem::GeomViewerItem(GeomObj* obj,const string& nm, 
			       CrowdMonitor* lck)
  :child(obj), name(nm), lock(lck)
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

GeomObj*
GeomViewerItem::clone()
{
  // Untested.
  return new GeomViewerItem(child, name, lock);
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

bool GeomViewerItem::saveobj(ostream& out, const string& format,
			     GeomSave* saveinfo)
{
    return child->saveobj(out, format, saveinfo);
}

} // End namespace SCIRun
