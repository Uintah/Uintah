
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

GeomContainer::GeomContainer(GeomObj* child)
: GeomObj(0), child(child)
{
}

GeomContainer::GeomContainer(const GeomContainer& copy)
: GeomObj(copy), child(copy.child)
{
}

GeomContainer::~GeomContainer()
{
}

void GeomContainer::get_bounds(BBox& bbox)
{
    child->get_bounds(bbox);
}

void GeomContainer::make_prims(Array1<GeomObj*>& free,
			       Array1<GeomObj*>& dontfree)
{
    child->make_prims(free, dontfree);
}

void GeomContainer::intersect(const Ray& ray, Material* matl,
			      Hit& hit)
{
    child->intersect(ray, matl, hit);
}
