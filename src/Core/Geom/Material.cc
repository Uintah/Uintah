/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

bool GeomMaterial::saveobj(ostream& out, const string& format,
			   GeomSave* saveinfo)
{
    cerr << "saveobj Material\n";
    if(format == "vrml" || format == "iv"){
	saveinfo->start_sep(out);
	saveinfo->start_node(out, "Material");
	saveinfo->indent(out);
	//Color& amb(matl->ambient);  // visual C++ hates constructors as reference initializers
	Color& amb = matl->ambient;
	out << "ambientColor " << amb.r() << " " << amb.g() << " " << amb.b() << "\n";

	saveinfo->indent(out);
	//Color& dif(matl->diffuse);
	Color& dif = matl->diffuse;
	out << "diffuseColor " << dif.r() << " " << dif.g() << " " << dif.b() << "\n";
	saveinfo->indent(out);
	//Color& spec(matl->specular);
	Color& spec = matl->specular;
	out << "specularColor " << spec.r() << " " << spec.g() << " " << spec.b() << "\n";
	saveinfo->indent(out);
	//Color& em(matl->emission);
	Color& em = matl->emission;
	out << "emissiveColor " << em.r() << " " << em.g() << " " << em.b() << "\n";
	saveinfo->indent(out);
	out << "shininess " << matl->shininess << "\n";
	saveinfo->indent(out);
	out << "transparency " << matl->transparency << "\n";
	saveinfo->end_node(out);
	if(!child_->saveobj(out, format, saveinfo))
	    return false;
	saveinfo->end_sep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_attr(out);
	//Color& spec(matl->specular);
	Color& spec = matl->specular;
	//Color& dif(matl->diffuse);
	Color& dif = matl->diffuse;

	saveinfo->indent(out);
	out << "Color [ " << dif.r() << " " << dif.g() << " " << dif.b() << " ]\n";
	saveinfo->indent(out);
	out << "Surface \"plastic\" \"Ka\" 0.0 \"Kd\" 1.0 \"Ks\" 1.0 \"roughness\" "
	    << 1.0 / matl->shininess << " \"specularcolor\" [ "
	    << spec.r() << " " << spec.g() << " " << spec.b() << " ]\n";

	if(!child_->saveobj(out, format, saveinfo))
	    return false;
	saveinfo->end_attr(out);
	return true;
    } else {
	NOT_FINISHED("GeomMaterial::saveobj");
	return false;
    }
}

} // End namespace SCIRun

// $Log

