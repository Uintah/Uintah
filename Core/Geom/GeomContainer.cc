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
#include <Core/Containers/String.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace SCIRun {

PersistentTypeID GeomContainer::type_id("GeomContainer", "GeomObj", 0);

GeomContainer::GeomContainer(GeomObj* child)
: GeomObj(), child(child)
{
}

GeomContainer::GeomContainer(const GeomContainer& copy)
: GeomObj(copy), child(copy.child->clone())
{
}

GeomContainer::~GeomContainer()
{
    if(child)
	delete child;
}

void GeomContainer::get_triangles( Array1<float> &v)
{
  if (child)
    child->get_triangles(v);
}

void GeomContainer::get_bounds(BBox& bbox)
{
    child->get_bounds(bbox);
}

#define GEOMCONTAINER_VERSION 1

void GeomContainer::io(Piostream& stream)
{
    stream.begin_class("GeomContainer", GEOMCONTAINER_VERSION);
    GeomObj::io(stream);
    Pio(stream, child);
    stream.end_class();
}

} // End namespace SCIRun
