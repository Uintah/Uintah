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


Persistent*
GeomViewerItem::maker()
{
  return scinew GeomViewerItem;
}

PersistentTypeID GeomViewerItem::type_id("GeomViewerItem", "GeomContainer",
					 GeomViewerItem::maker);

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
  GeomContainer(0),
  crowd_lock_(0)
{
}

GeomViewerItem::GeomViewerItem(GeomHandle obj,const string& nm, 
			       CrowdMonitor* lck) :
  GeomContainer(obj),
  name_(nm),
  crowd_lock_(lck)
{
  if (!crowd_lock_)
  {
    child_ = scinew GeomBBoxCache(obj);
  }
}

GeomObj*
GeomViewerItem::clone()
{
  // Untested.
  return scinew GeomViewerItem(child_, name_, crowd_lock_);
}

#define GeomViewerITEM_VERSION 1

void
GeomViewerItem::io(Piostream& stream)
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


} // End namespace SCIRun
