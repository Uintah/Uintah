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

GeomViewerItem::GeomViewerItem() :
  child_(0),
  crowd_lock_(0)
{
}

GeomViewerItem::GeomViewerItem(GeomObjHandle obj,const string& nm, 
			       CrowdMonitor* lck)
  :child_(obj), name_(nm), crowd_lock_(lck)
{
    if (!crowd_lock_)
	child_ = scinew GeomBBoxCache(obj);
}

GeomViewerItem::~GeomViewerItem()
{
}

void GeomViewerItem::get_triangles( Array1<float> &v)
{
  if ( child_)
    child_->get_triangles(v);
}

GeomObj*
GeomViewerItem::clone()
{
  // Untested.
  return scinew GeomViewerItem(child_, name_, crowd_lock_);
}

void GeomViewerItem::reset_bbox()
{
  if (child_)
  {
    child_->reset_bbox();
  }
}

void GeomViewerItem::get_bounds(BBox& box)
{
  if (child_)
  {
    child_->get_bounds(box);
  }
}

#define GeomViewerITEM_VERSION 1

void GeomViewerItem::io(Piostream& stream)
{

    stream.begin_class("GeomViewerItem", GeomViewerITEM_VERSION);
    int have_lock;
    if(stream.writing())
	have_lock=crowd_lock_?1:0;
    Pio(stream, have_lock);
    if(stream.reading())
	if(have_lock)
	    crowd_lock_ = scinew CrowdMonitor("GeomViewerItem crowd monitor");
	else
	    crowd_lock_ = 0;
    Pio(stream, name_);
    Pio(stream, child_);
    stream.end_class();
}

bool GeomViewerItem::saveobj(ostream& out, const string& format,
			     GeomSave* saveinfo)
{
  if (child_)
  {
    return child_->saveobj(out, format, saveinfo);
  }
  return true;
}

} // End namespace SCIRun
