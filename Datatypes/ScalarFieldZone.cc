
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
    NOT_FINISHED("ScalarFieldZone::compute_bounds");
}

void ScalarFieldZone::compute_minmax()
{
    NOT_FINISHED("ScalarFieldZone::compute_minmax");
}

Vector ScalarFieldZone::gradient(const Point&)
{
    NOT_FINISHED("ScalarFieldZone::gradient");
    return Vector(0,0,0);
}

int ScalarFieldZone::interpolate(const Point&, double&, double epsilon1,
				 double epsilon2)
{
    NOT_FINISHED("ScalarFieldZone::interpolate");
    return 0;
}

int ScalarFieldZone::interpolate(const Point&, double&, int& ix,
				 double epsilon1, double epsilon2)
{
    NOT_FINISHED("ScalarFieldZone::interpolate");
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

