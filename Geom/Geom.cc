
/*
 *  Geom.cc: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Geom.h>
#include <Geometry/Vector.h>
#include <iostream.h>

GeomObj::GeomObj(int lit)
: lit(lit)
{
}

GeomObj::GeomObj(const GeomObj& copy)
: lit(copy.lit)
{
}

GeomObj::~GeomObj()
{
}

void GeomObj::reset_bbox()
{
    // Nothing to do, by default.
}

Vector GeomObj::normal(const Point&)
{
    cerr << "ERROR: GeomObj::normal() shouldn't get called!!!\n";
    return Vector(0,0,1);
}
