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
 *  ViewGeom.cc:
 *
 *   Department of Computer Science
 *   University of Utah
 *   Date: November 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <sci_defs/ogl_defs.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Geom/GeomViewerItem.h>
#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Geom/BBoxCache.h>

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
  return scinew GeomViewerItem(child_, name_, 0);//crowd_lock_);
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

void
GeomViewerItem::fbpick_draw(DrawInfoOpenGL* di, Material *m, double time)
{
  std::cerr << "calling child_ fbpick_draw" << std::endl;
  child_->fbpick_draw(di,m,time);
}

void
GeomViewerItem::draw(DrawInfoOpenGL* di, Material *m, double time)
{

  // Here we need to query the ViewWindow with our name and give it our
  // di so it can change things if they need to be.
  //  di->viewwindow->setDI(di, name_);

  BBox bb;
  child_->get_bounds(bb);
  if (!(di->show_bbox_ && bb.valid()))
  {
    child_->draw(di,m,time);
  }
  else
  {
    const Point &min(bb.min());
    const Point &max(bb.max());

    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);

    glDisable(GL_LIGHTING);     

    glBegin(GL_QUADS);

    //top
    glColor4f(0.0, 0.0, 1.0, 0.4);
    glVertex3d(max.x(),min.y(),max.z());
    glVertex3d(max.x(),max.y(),max.z());
    glVertex3d(min.x(),max.y(),max.z());
    glVertex3d(min.x(),min.y(),max.z());

    //bottom
    glColor4f(0.0, 0.0, 1.0, 0.2);
    glVertex3d(max.x(),max.y(),min.z());
    glVertex3d(max.x(),min.y(),min.z());
    glVertex3d(min.x(),min.y(),min.z());
    glVertex3d(min.x(),max.y(),min.z());

    //right
    glColor4f(1.0, 0.0, 0.0, 0.4);
    glVertex3d(max.x(),min.y(),min.z());
    glVertex3d(max.x(),max.y(),min.z());
    glVertex3d(max.x(),max.y(),max.z());
    glVertex3d(max.x(),min.y(),max.z());

    //left
    glColor4f(1.0, 0.0, 0.0, 0.2);
    glVertex3d(min.x(),min.y(),max.z());
    glVertex3d(min.x(),max.y(),max.z());
    glVertex3d(min.x(),max.y(),min.z());
    glVertex3d(min.x(),min.y(),min.z());

    //top
    glColor4f(0.0, 1.0, 0.0, 0.4);
    glVertex3d(min.x(),max.y(),max.z());
    glVertex3d(max.x(),max.y(),max.z());
    glVertex3d(max.x(),max.y(),min.z());
    glVertex3d(min.x(),max.y(),min.z());

    //bottom
    glColor4f(0.0, 1.0, 0.0, 0.2);
    glVertex3d(min.x(),min.y(),min.z());
    glVertex3d(max.x(),min.y(),min.z());
    glVertex3d(max.x(),min.y(),max.z());
    glVertex3d(min.x(),min.y(),max.z());

    glEnd();
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);
  }
}



} // End namespace SCIRun
