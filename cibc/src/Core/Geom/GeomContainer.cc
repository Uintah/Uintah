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
 *  Container.cc: Base class for container objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomContainer.h>

#include <iostream>
namespace SCIRun {

using std::cerr;
using std::endl;
using std::ostream;

PersistentTypeID GeomContainer::type_id("GeomContainer", "GeomObj", 0);

GeomContainer::GeomContainer(GeomHandle child) :
  GeomObj(),
  child_(child)
{
}


GeomContainer::GeomContainer(const GeomContainer& copy) :
  GeomObj(copy),
  child_(copy.child_)
{
}


GeomContainer::~GeomContainer()
{
}


void
GeomContainer::get_triangles( Array1<float> &v)
{
  if (child_.get_rep())
    child_->get_triangles(v);
}


void
GeomContainer::get_bounds(BBox& bbox)
{
  if (child_.get_rep())
    child_->get_bounds(bbox);
}

void
GeomContainer::reset_bbox()
{
  if (child_.get_rep())
    child_->reset_bbox();
}

#define GEOMCONTAINER_VERSION 1

void
GeomContainer::io(Piostream& stream)
{
    stream.begin_class("GeomContainer", GEOMCONTAINER_VERSION);
    GeomObj::io(stream);
    Pio(stream, child_);
    stream.end_class();
}


} // End namespace SCIRun
