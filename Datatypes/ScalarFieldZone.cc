
/*
 *  ScalarFieldZone.h: A compound scalar field type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ScalarFieldZone.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>

static Persistent* maker()
{
    return scinew ScalarFieldZone(0);
}

PersistentTypeID ScalarFieldZone::type_id("ScalarFieldZone", "ScalarField", maker);

ScalarFieldZone::ScalarFieldZone(int nzones)
: ScalarField(Zones), zones(nzones)
{
}

ScalarFieldZone::~ScalarFieldZone()
{
}

ScalarField* ScalarFieldZone::clone()
{
    NOT_FINISHED("ScalarFieldZone::clone");
    return 0;
}

void ScalarFieldZone::compute_bounds()
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

void ScalarFieldZone::compute_minmax()
{
    if(zones.size()==0)
	return;
    zones[0]->get_minmax(data_min, data_max);
    for(int i=1;i<zones.size();i++){
	double  zmin, zmax;
	zones[i]->get_minmax(zmin, zmax);
	data_min=Min(data_min, zmin);
	data_max=Max(data_max, zmax);
    }
}

Vector ScalarFieldZone::gradient(const Point&)
{
    NOT_FINISHED("ScalarFieldZone::gradient");
    return Vector(0,0,0);
}

int ScalarFieldZone::interpolate(const Point& p, double& v, double epsilon1,
				 double epsilon2)
{
    for(int i=0;i<zones.size();i++){
	if(zones[i]->interpolate(p, v, epsilon1, epsilon2))
	    return 1;
    }
    return 0;
}

int ScalarFieldZone::interpolate(const Point& p, double& v, int& cache,
				 double epsilon1, double epsilon2)
{
    for(int i=0;i<zones.size();i++){
	if(zones[i]->interpolate(p, v, cache, epsilon1, epsilon2))
	    return 1;
    }
    return 0;
}

#define SCALARFIELDZONE_VERSION 1

void ScalarFieldZone::io(Piostream& stream)
{
    /* int version=*/stream.begin_class("ScalarFieldZone", SCALARFIELDZONE_VERSION);
    Pio(stream, zones);
    stream.end_class();
}

void ScalarFieldZone::get_boundary_lines(Array1<Point>& lines)
{
    for(int i=0;i<zones.size();i++)
	zones[i]->get_boundary_lines(lines);
}

