
/*
 *  GeomRaytracer.h: Information for Ray Tracing
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/GeomRaytracer.h>

Hit::Hit()
: _prim(0)
{
}

Hit::~Hit()
{
}

double Hit::t() const
{
    return _t;
}

int Hit::hit() const
{
    return _prim!=0;
}

GeomObj* Hit::prim() const
{
    return _prim;
}

MaterialHandle Hit::matl() const
{
    return _matl;
}

void Hit::hit(double new_t, GeomObj* new_prim, const MaterialHandle& new_matl)
{
    if(new_t > 1.e-6 && (new_t < _t || !_prim)){
	_t=new_t;
	_prim=new_prim;
	_matl=new_matl;
    }
}

OcclusionData::OcclusionData(GeomObj* hit_prim, Raytracer* raytracer,
			     int level, View* view)
: hit_prim(hit_prim), raytracer(raytracer), level(level), view(view)
{
}
