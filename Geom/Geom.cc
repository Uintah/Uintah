
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

Material::Material(const Color& ambient, const Color& diffuse,
		   const Color& specular, double shininess)
: ref_cnt(0), ambient(ambient), diffuse(diffuse), specular(specular),
  shininess(shininess), emission(0,0,0)
{
}

GeomObj::GeomObj(int lit)
: lit(lit), matl(0), pick(0)
{
}

GeomObj::GeomObj(const GeomObj& copy)
: matl(copy.matl), pick(copy.pick)
{
}

GeomObj::~GeomObj()
{
}

void GeomObj::set_pick(GeomPick* _pick)
{
    pick=_pick;
}

GeomPick* GeomObj::get_pick()
{
    return pick;
}

void GeomObj::set_matl(const MaterialHandle& _matl)
{
    matl=_matl;
}


