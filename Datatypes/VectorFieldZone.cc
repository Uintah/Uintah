
/*
 *  VectorFieldZone.h: A compound Vector field type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/VectorFieldZone.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>

static Persistent* maker()
{
    return scinew VectorFieldZone(0);
}

PersistentTypeID VectorFieldZone::type_id("VectorFieldZone", "VectorField", maker);

VectorFieldZone::VectorFieldZone(int nzones)
: VectorField(Zones), zones(nzones)
{
}

VectorFieldZone::~VectorFieldZone()
{
}

VectorField* VectorFieldZone::clone()
{
    NOT_FINISHED("VectorFieldZone::clone");
    return 0;
}

void VectorFieldZone::compute_bounds()
{
    if(zones.size()==0)
	return;
    zones[0]->get_bounds(bmin, bmax);
    for(int i=1;i<zones.size();i++){
	Point min, max;
	zones[i]->get_bounds(min, max);
	bmin=Min(min, bmin);
	bmax=Max(max, bmax);
    }
}

int VectorFieldZone::interpolate(const Point& p, Vector& v)
{
    for(int i=0;i<zones.size();i++){
	if(zones[i]->interpolate(p, v))
	    return 1;
    }
    return 0;
}

int VectorFieldZone::interpolate(const Point& p, Vector& v, int& cache)
{
    for(int i=0;i<zones.size();i++){
	if(zones[i]->interpolate(p, v, cache))
	    return 1;
    }
    return 0;
}

#define VECTORFIELDZONE_VERSION 1

void VectorFieldZone::io(Piostream& stream)
{
    /* int version=*/stream.begin_class("VectorFieldZone", VECTORFIELDZONE_VERSION);
    Pio(stream, zones);
    stream.end_class();
}

void VectorFieldZone::get_boundary_lines(Array1<Point>& lines)
{
    for(int i=0;i<zones.size();i++)
	zones[i]->get_boundary_lines(lines);
}

