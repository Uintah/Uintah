/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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


#include <Core/Geom/GeomOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/BBoxCache.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Modules/Render/ViewGeom.h>
#include <Dataflow/Modules/Render/ViewWindow.h>

#include <iostream>
using std::ostream;

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
