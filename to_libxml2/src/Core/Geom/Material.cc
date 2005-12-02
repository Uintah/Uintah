/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



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

#include <Core/Geom/Material.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;
//#include <Core/Containers/LockingHandle.h>

namespace SCIRun {


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
  : ref_cnt(0), lock("Material mutex"), ambient(0,0,0), diffuse(0,0,0),
    specular(0,0,0), shininess(0), emission(0,0,0), reflectivity(0.5),
    transparency(0), refraction_index(1)
{
}

Material::Material(const Color& ambient, const Color& diffuse,
		   const Color& specular, double shininess)
  : ref_cnt(0), lock("Material mutex"), ambient(ambient), diffuse(diffuse),
    specular(specular), shininess(shininess),
    emission(0,0,0), reflectivity(0.5),
    transparency(0), refraction_index(1)
{
}

Material::Material(const Color& diffuse)
  : ref_cnt(0), lock("Material mutex"), diffuse(diffuse), emission(0,0,0),
    reflectivity(0.5), transparency(0), refraction_index(1)
{
    ambient=Color(.2,.2,.2);
    specular=Color(.8,.8,.8);
    shininess=20;
}

Material::Material(const Material& copy)
: ref_cnt(0), lock("Material mutex"), ambient(copy.ambient),
  diffuse(copy.diffuse),
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

GeomMaterial::GeomMaterial(GeomHandle obj, const MaterialHandle& matl)
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

#define GEOMMATERIAL_VERSION 1

void GeomMaterial::io(Piostream& stream)
{

    stream.begin_class("GeomMaterial", GEOMMATERIAL_VERSION);
    GeomContainer::io(stream);
    Pio(stream, matl);
    stream.end_class();
}

} // End namespace SCIRun

// $Log

