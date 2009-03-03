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
 * ViewWindowClipFrame.cc: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/ViewWindowClipFrame.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/Persistent/PersistentSTL.h>

#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_ViewWindowClipFrame()
{
    return scinew ViewWindowClipFrame;
}

PersistentTypeID 
ViewWindowClipFrame::type_id("ViewWindowClipFrame", "GeomObj", 
                             make_ViewWindowClipFrame);


ViewWindowClipFrame::ViewWindowClipFrame() :
  GeomObj(),
  center_(Point(0,0,0)),
  normal_(Vector(1,0,0)),
  width_(2.0),
  height_(2.0),
  scale_(1.0),
  verts_(5, Point(0,0,0)),
  corners_(4),
  edges_(4)
{
  unsigned int i;
  for(i = 0; i < 4; i++){
    corners_[i] = scinew GeomSphere;
    edges_[i] = scinew GeomCylinder;
  }
}

#define VIEWWINDOWCLIPFRAME_VERSION 1

void ViewWindowClipFrame::io(Piostream& stream)
{

    stream.begin_class("ViewWindowClipFrame", VIEWWINDOWCLIPFRAME_VERSION);
    GeomObj::io(stream);
    Pio(stream, center_);
    Pio(stream, normal_);
    Pio(stream, width_);
    Pio(stream, height_);
    Pio(stream, scale_);
    stream.end_class();
}

GeomObj*
ViewWindowClipFrame::clone()
{
  return scinew ViewWindowClipFrame(*this);
}


ViewWindowClipFrame::ViewWindowClipFrame(const ViewWindowClipFrame& copy) :
  GeomObj(copy), 
  center_(copy.center_),
  normal_(copy.normal_),
  width_(copy.width_),
  height_(copy.height_),
  scale_(copy.scale_),
  verts_(copy.verts_),
  corners_( copy.corners_),
  edges_(copy.edges_)
{}

ViewWindowClipFrame::~ViewWindowClipFrame()
{
  unsigned int i;
  for(i = 0; i < 4; i++){
    delete corners_[i];
    delete edges_[i];
  }
}

void
ViewWindowClipFrame::get_bounds(BBox& bb)
{
  unsigned int i;
  for(i = 0; i < edges_.size(); i++){
    edges_[i]->get_bounds(bb);
    corners_[i]->get_bounds(bb);
  }
}
void
ViewWindowClipFrame::SetPosition(Point c, Vector n)
{
  set_position(c,n);
  adjust();
}


void
ViewWindowClipFrame::SetSize(double w, double h)
{
  set_size(w,h);
  adjust();
}


void
ViewWindowClipFrame::SetScale(double s)
{
  set_scale(s);
  adjust();
}

void
ViewWindowClipFrame::Set(Point c, Vector n,
                         double w, double h,
                         double s)
{
  set_position(c,n);
  set_size(w,h);
  set_scale(s);
  adjust();
}

void
ViewWindowClipFrame::adjust()
{
  unsigned int i;

  // establish the corners

  Vector axis1, axis2, right, down;
  normal_.find_orthogonal(axis1, axis2);
  right = axis1*width_*0.5;
  down = axis2*height_*0.5;
  
  verts_[0] = center_ - right + down;
  verts_[1] = center_ + right + down;
  verts_[2] = center_ + right - down;
  verts_[3] = center_ - right - down;
  verts_[4] = verts_[0];
  
  for( i = 0; i < edges_.size(); i++){
    edges_[i]->move( verts_[i], verts_[i+1], scale_);
    corners_[i]->move( verts_[i], scale_ );
  }
}


} // End namespace SCIRun

