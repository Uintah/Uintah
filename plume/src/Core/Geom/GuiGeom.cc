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
#include <Core/GuiInterface/GuiInterface.h>
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
  ctx->getInterface()->lock();
  Color c(r.get(), g.get(), b.get());
  ctx->getInterface()->unlock();
  return c;
}

void GuiColor::set(const Color& p)
{
  ctx->getInterface()->lock();
  r.set(p.r());
  g.set(p.g());
  b.set(p.b());
  ctx->getInterface()->unlock();
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


