
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
#include <Malloc/Allocator.h>

static Persistent* make_Material()
{
    return scinew Material;
}

PersistentTypeID Material::type_id("Material", "Persistent", make_Material);

Persistent* make_GeomMaterial()
{
    return new GeomMaterial(0, MaterialHandle(0));
}

PersistentTypeID GeomMaterial::type_id("GeomMaterial", "GeomObj", make_GeomMaterial);

Material::Material()
: ref_cnt(0), ambient(0,0,0), diffuse(0,0,0), specular(0,0,0),
  shininess(0), emission(0,0,0), reflectivity(0.5),
  transparency(0), refraction_index(1)
{
}

Material::Material(const Color& ambient, const Color& diffuse,
		   const Color& specular, double shininess)
: ref_cnt(0), ambient(ambient), diffuse(diffuse), specular(specular),
  shininess(shininess), emission(0,0,0), reflectivity(0.5),
  transparency(0), refraction_index(1)
{
}

Material::Material(const Material& copy)
: ref_cnt(0), ambient(copy.ambient), diffuse(copy.diffuse),
  specular(copy.specular), shininess(copy.shininess),
  emission(copy.emission), reflectivity(copy.reflectivity),
  transparency(copy.transparency),
  refraction_index(copy.refraction_index)
{
}

Material::~Material()
{
}

Material& Material::operator=(const Material& copy)
{
   ambient=copy.ambient;
   diffuse=copy.diffuse;
   specular=copy.specular;
   shininess=copy.shininess;
   emission=copy.emission;
   reflectivity=copy.reflectivity;
   transparency=copy.transparency;
   refraction_index=copy.refraction_index;
   return *this;
}

Material* Material::clone()
{
    return scinew Material(*this);
}

#define MATERIAL_VERSION 1

void Material::io(Piostream& stream)
{
    /* int version= */stream.begin_class("Material", MATERIAL_VERSION);
    Pio(stream, ambient);
    Pio(stream, diffuse);
    Pio(stream, specular);
    Pio(stream, shininess);
    Pio(stream, emission);
    Pio(stream, reflectivity);
    Pio(stream, transparency);
    Pio(stream, refraction_index);
}

GeomMaterial::GeomMaterial(GeomObj* obj, const MaterialHandle& matl)
: GeomContainer(obj), matl(matl)
{
}

GeomMaterial::GeomMaterial(const GeomMaterial& copy)
: GeomContainer(copy), matl(copy.matl)
{
}

void GeomMaterial::setMaterial(const MaterialHandle& copy) 
{
    matl=copy;
}

MaterialHandle GeomMaterial::getMaterial()
{
    return matl;
}

GeomMaterial::~GeomMaterial()
{
}

GeomObj* GeomMaterial::clone()
{
    return scinew GeomMaterial(*this);
}

void GeomMaterial::make_prims(Array1<GeomObj*>& free,
			      Array1<GeomObj*>& dontfree)
{
    child->make_prims(free, dontfree);
}

void GeomMaterial::intersect(const Ray& ray, Material* /* old_matl */,
			     Hit& hit)
{
    child->intersect(ray, matl.get_rep(), hit);
}

#define GEOMMATERIAL_VERSION 1

void GeomMaterial::io(Piostream& stream)
{
    stream.begin_class("GeomMaterial", GEOMMATERIAL_VERSION);
    GeomContainer::io(stream);
    Pio(stream, matl);
    stream.end_class();
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template class LockingHandle<Material>;

#include <Classlib/Array1.cc>
template class Array1<MaterialHandle>;

template void Pio(Piostream&, Array1<MaterialHandle>&);
template void Pio(Piostream&, MaterialHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/LockingHandle.cc>

static void _dummy_(Piostream& p1, MaterialHandle& p2)
{
    Pio(p1, p2);
}

#endif
#endif

