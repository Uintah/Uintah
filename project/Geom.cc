
/*
 *  Geom.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom.h>

GeomObj::GeomObj()
: matl(0)
{
}

GeomObj::~GeomObj()
{
#if 0
    if(matl)
	delete matl;
#endif
}

ObjGroup::ObjGroup()
: objs(0, 100)
{
}

ObjGroup::~ObjGroup()
{
}

void ObjGroup::add(GeomObj* obj)
{
    objs.add(obj);
}

int ObjGroup::size()
{
    return objs.size();
}

Triangle::Triangle(const Point& p1, const Point& p2, const Point& p3)
: p1(p1), p2(p2), p3(p3)
{
}

Triangle::~Triangle()
{
}
