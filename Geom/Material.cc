
/*
 *  Material.cc: Material properties for Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Material.h>
#include <Classlib/String.h>

static Persistent* make_Material()
{
    return new Material;
}

PersistentTypeID Material::type_id("Material", "Persistent", make_Material);

Material::Material()
: ref_cnt(0)
{
}

Material::Material(const Color& ambient, const Color& diffuse,
		   const Color& specular, double shininess)
: ref_cnt(0), ambient(ambient), diffuse(diffuse), specular(specular),
  shininess(shininess), emission(0,0,0)
{
}

Material::Material(const Material& copy)
: ref_cnt(0), ambient(copy.ambient), diffuse(copy.diffuse),
  specular(copy.specular), shininess(copy.shininess),
  emission(copy.emission)
{
}

Material::~Material()
{
}

Material* Material::clone()
{
    return new Material(*this);
}

#define MATERIAL_VERSION 1

void Material::io(Piostream& stream)
{
    int version=stream.begin_class("Material", MATERIAL_VERSION);
    Pio(stream, ambient);
    Pio(stream, diffuse);
    Pio(stream, specular);
    Pio(stream, shininess);
    Pio(stream, emission);
}
