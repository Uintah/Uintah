
/*
 *  GuiGeom.cc: Interface to TCL variables for Geom stuff
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GuiGeom.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/Material.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

GuiColor::GuiColor(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl), r("r", str(), tcl), g("g", str(), tcl),
  b("b", str(), tcl)
{
}

GuiColor::~GuiColor()
{
}

void GuiColor::reset() {
  r.reset();
  g.reset();
  b.reset();
}

Color GuiColor::get()
{
    return Color(r.get(), g.get(), b.get());
}

void GuiColor::set(const Color& p)
{
    r.set(p.r());
    g.set(p.g());
    b.set(p.b());
}

void GuiColor::emit(ostream& out)
{
    r.emit(out);
    g.emit(out);
    b.emit(out);
}

GuiMaterial::GuiMaterial(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl), ambient("ambient", str(), tcl),
  diffuse("diffuse", str(), tcl), specular("specular", str(), tcl),
  shininess("shininess", str(), tcl), emission("emission", str(), tcl),
  reflectivity("reflectivity", str(), tcl),
  transparency("transparency", str(), tcl),
  refraction_index("refraction_index", str(), tcl)
{
}

GuiMaterial::~GuiMaterial()
{
}

void GuiMaterial::reset() {
  ambient.reset();
  diffuse.reset();
  specular.reset();
  shininess.reset();
  emission.reset();
  reflectivity.reset();
  transparency.reset();
  refraction_index.reset();
}

Material GuiMaterial::get()
{
    Material m(ambient.get(), diffuse.get(), specular.get(), shininess.get());
    m.emission=emission.get();
    m.reflectivity=reflectivity.get();
    m.transparency=transparency.get();
    m.refraction_index=refraction_index.get();
    return m;
}

void GuiMaterial::set(const Material& m)
{
    ambient.set(m.ambient);
    diffuse.set(m.diffuse);
    specular.set(m.specular);
    shininess.set(m.shininess);
    emission.set(m.emission);
    reflectivity.set(m.reflectivity);
    transparency.set(m.transparency);
    refraction_index.set(m.refraction_index);
}

void GuiMaterial::emit(ostream& out)
{
    ambient.emit(out);
    diffuse.emit(out);
    specular.emit(out);
    shininess.emit(out);
    emission.emit(out);
    reflectivity.emit(out);
    transparency.emit(out);
    refraction_index.emit(out);
}

} // End namespace SCIRun

