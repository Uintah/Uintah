
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
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geom/Save.h>
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

Material::Material(const Color& diffuse)
: ref_cnt(0), diffuse(diffuse), emission(0,0,0), reflectivity(0.5),
  transparency(0), refraction_index(1)
{
    ambient=Color(.2,.2,.2);
    specular=Color(.8,.8,.8);
    shininess=20;
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

bool GeomMaterial::saveobj(ostream& out, const clString& format,
			   GeomSave* saveinfo)
{
    cerr << "saveobj Material\n";
    if(format == "vrml" || format == "iv"){
	saveinfo->start_sep(out);
	saveinfo->start_node(out, "Material");
	saveinfo->indent(out);
	Color& amb(matl->ambient);
	out << "ambientColor " << amb.r() << " " << amb.g() << " " << amb.b() << "\n";

	saveinfo->indent(out);
	Color& dif(matl->diffuse);
	out << "diffuseColor " << dif.r() << " " << dif.g() << " " << dif.b() << "\n";
	saveinfo->indent(out);
	Color& spec(matl->specular);
	out << "specularColor " << spec.r() << " " << spec.g() << " " << spec.b() << "\n";
	saveinfo->indent(out);
	Color& em(matl->emission);
	out << "emissiveColor " << em.r() << " " << em.g() << " " << em.b() << "\n";
	saveinfo->indent(out);
	out << "shininess " << matl->shininess << "\n";
	saveinfo->indent(out);
	out << "transparency " << matl->transparency << "\n";
	saveinfo->end_node(out);
	if(!child->saveobj(out, format, saveinfo))
	    return false;
	saveinfo->end_sep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_attr(out);
	Color& spec(matl->specular);
	Color& dif(matl->diffuse);

	saveinfo->indent(out);
	out << "Color [ " << dif.r() << " " << dif.g() << " " << dif.b() << " ]\n";
	saveinfo->indent(out);
	out << "Surface \"plastic\" \"Ka\" 0.0 \"Kd\" 1.0 \"Ks\" 1.0 \"roughness\" "
	    << 1.0 / matl->shininess << " \"specularcolor\" [ "
	    << spec.r() << " " << spec.g() << " " << spec.b() << " ]\n";

	if(!child->saveobj(out, format, saveinfo))
	    return false;
	saveinfo->end_attr(out);
	return true;
    } else {
	NOT_FINISHED("GeomMaterial::saveobj");
	return false;
    }
}

