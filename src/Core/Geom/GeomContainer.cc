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
