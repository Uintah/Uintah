
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

#include <Geom/Container.h>
#include <Classlib/String.h>

PersistentTypeID GeomContainer::type_id("GeomContainer", "GeomObj", 0);

GeomContainer::GeomContainer(GeomObj* child)
: GeomObj(), child(child)
{
    if(child)
	child->set_parent(this);
}

GeomContainer::GeomContainer(const GeomContainer& copy)
: GeomObj(copy), child(copy.child->clone())
{
    if(child)
	child->set_parent(this);
}

GeomContainer::~GeomContainer()
{
    if(child)
	delete child;
}

void GeomContainer::get_bounds(BBox& bbox)
{
    child->get_bounds(bbox);
}

void GeomContainer::get_bounds(BSphere& bsphere)
{
    child->get_bounds(bsphere);
}

void GeomContainer::make_prims(Array1<GeomObj*>& free,
			       Array1<GeomObj*>& dontfree)
{
    child->make_prims(free, dontfree);
}

void GeomContainer::preprocess()
{
    child->preprocess();
}

void GeomContainer::intersect(const Ray& ray, Material* matl,
			      Hit& hit)
{
    child->intersect(ray, matl, hit);
}

#define GEOMCONTAINER_VERSION 1

void GeomContainer::io(Piostream& stream)
{
    stream.begin_class("GeomContainer", GEOMCONTAINER_VERSION);
    GeomObj::io(stream);
    Pio(stream, child);
    stream.end_class();
}
