
/*
 *  Switch.h:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */
#include <Geom/Switch.h>

GeomSwitch::GeomSwitch(GeomObj* obj, int state)
: GeomContainer(obj), state(state)
{
}

GeomSwitch::GeomSwitch(const GeomSwitch& copy)
: GeomContainer(copy), state(copy.state)
{
}

GeomSwitch::~GeomSwitch()
{
}

GeomObj* GeomSwitch::clone()
{
    return new GeomSwitch(*this);
}

void GeomSwitch::set_state(int st)
{
   state=st;
}

int GeomSwitch::get_state()
{
   return state;
}

void GeomSwitch::get_bounds(BBox& bbox)
{
   if(state)child->get_bounds(bbox);
}


void GeomSwitch::get_bounds(BSphere& bs)
{
   if(state)child->get_bounds(bs);
}

void GeomSwitch::make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree)
{
   if(state)child->make_prims(free, dontfree);
}


void GeomSwitch::preprocess()
{
   if(state)child->preprocess();
}


void GeomSwitch::intersect(const Ray& ray, Material* matl, Hit& hit)
{
   if(state)child->intersect(ray, matl, hit);
}

GeomTimeSwitch::GeomTimeSwitch(GeomObj* obj, double tbeg, double tend)
: GeomContainer(obj), tbeg(tbeg), tend(tend)
{
}

GeomTimeSwitch::GeomTimeSwitch(const GeomTimeSwitch& copy)
: GeomContainer(copy), tbeg(copy.tbeg), tend(copy.tend)
{
}

GeomTimeSwitch::~GeomTimeSwitch()
{
}

GeomObj* GeomTimeSwitch::clone()
{
    return new GeomTimeSwitch(*this);
}
