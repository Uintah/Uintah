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

#include <Core/Geom/GuiGeom.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
using namespace SCIRun;

GuiColor::GuiColor(GuiContext* ctx)
  : GuiVar(ctx), r(ctx->subVar("r")), g(ctx->subVar("g")),
    b(ctx->subVar("b"))
{
}

GuiColor::~GuiColor()
{
}

Color GuiColor::get()
{
  ctx->lock();
  Color c(r.get(), g.get(), b.get());
  ctx->unlock();
  return c;
}

void GuiColor::set(const Color& p)
{
  ctx->lock();
  r.set(p.r());
  g.set(p.g());
  b.set(p.b());
  ctx->unlock();
}

GuiMaterial::GuiMaterial(GuiContext* ctx)
: GuiVar(ctx), ambient(ctx->subVar("ambient")),
  diffuse(ctx->subVar("diffuse")), specular(ctx->subVar("specular")),
  shininess(ctx->subVar("shininess")), emission(ctx->subVar("emission")),
  reflectivity(ctx->subVar("reflectivity")),
  transparency(ctx->subVar("transparency")),
  refraction_index(ctx->subVar("refraction_index"))
{
}

GuiMaterial::~GuiMaterial()
{
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


