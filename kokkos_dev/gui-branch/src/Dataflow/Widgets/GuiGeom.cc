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

#include <Dataflow/Widgets/GuiGeom.h>
#include <Core/GuiInterface/TCLArgs.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/Material.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

GuiColor::GuiColor(const string& name, Part* part)
: GuiVar(name, part),  r(name+"_r", part), g(name+"_g", part),  b(name+"_b", part)
{
}

GuiColor::GuiColor(const string& name, const string& id, Part* part)
: GuiVar(name, part),  r(name+"_r", part), g(name+"_g", part),  b(name+"_b", part)
{
}

GuiColor::~GuiColor()
{
}

void GuiColor::reset() {
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

void GuiColor::emit(ostream& out, string& midx)
{
    r.emit(out, midx);
    g.emit(out, midx);
    b.emit(out, midx);
}

GuiMaterial::GuiMaterial(const string& name, const string& id, Part* part)
: GuiVar(name, part), ambient(name+"_ambient", part),
  diffuse(name+"_diffuse", part), specular(name+"_specular", part),
  shininess(name+"_shininess", part), emission(name+"_emission", part),
  reflectivity(name+"_reflectivity", part),
  transparency(name+"_transparency", part),
  refraction_index(name+"_refraction_index", part)
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

void GuiMaterial::emit(ostream& out, string& midx)
{
    ambient.emit(out, midx);
    diffuse.emit(out, midx);
    specular.emit(out, midx);
    shininess.emit(out, midx);
    emission.emit(out, midx);
    reflectivity.emit(out, midx);
    transparency.emit(out, midx);
    refraction_index.emit(out, midx);
}

} // End namespace SCIRun

