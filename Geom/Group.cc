
/*
 *  Group.cc:  Groups of GeomObj's
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Group.h>

GeomGroup::GeomGroup()
: GeomObj(0), objs(0, 100)
{
}

GeomGroup::GeomGroup(const GeomGroup& copy)
: GeomObj(copy), bb(copy.bb)
{
    objs.grow(copy.objs.size());
    for(int i=0;i<objs.size();i++){
	GeomObj* cobj=copy.objs[i];
	objs[i]=cobj->clone();
    }
}

GeomGroup::~GeomGroup()
{
    for(int i=0;i<objs.size();i++)
	delete objs[i];
}

void GeomGroup::add(GeomObj* obj)
{
    objs.add(obj);
}

int GeomGroup::size()
{
    return objs.size();
}

GeomObj* GeomGroup::clone()
{
    return new GeomGroup(*this);
}

void GeomGroup::get_bounds(BBox& in_bb)
{
    if(!bb.valid()){
	for(int i=0;i<objs.size();i++)
	    objs[i]->get_bounds(bb);
    }
    if(bb.valid())
	in_bb.extend(bb);
}

void GeomGroup::make_prims(Array1<GeomObj*>& free,
			 Array1<GeomObj*>& dontfree)
{
    for(int i=0;i<objs.size();i++)
	objs[i]->make_prims(free, dontfree);
}

void GeomGroup::reset_bbox()
{
    for(int i=0;i<objs.size();i++)
	objs[i]->reset_bbox();
    bb.reset();
}
