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
 *  GuiView.cc  Structure that provides for easy access of view information.
 *              The view information is interactively provided by the user.
 *
 *  Written by:
 *   Steven Parker
 *   Department of Computer Science
 *   University of Utah
 *
 *   separated from the Viewer code by me (Aleksandra)
 *   in May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Core/GuiInterface/GuiContext.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geom/GuiView.h>
using namespace SCIRun;

GuiView::GuiView(GuiContext* ctx)
: GuiVar(ctx), eyep(ctx->subVar("eyep")),
  lookat(ctx->subVar("lookat")), up(ctx->subVar("up")),
  fov(ctx->subVar("fov")), eyep_offset(ctx->subVar("eyep_offset"))
{
}

GuiView::~GuiView()
{
}

View
GuiView::get()
{
  ctx->lock();
  View v(eyep.get(), lookat.get(), up.get(), fov.get());
  ctx->unlock();
  return v;
}

void
GuiView::set(const View& view)
{
  ctx->lock();
  eyep.set(view.eyep());
  lookat.set(view.lookat());
  up.set(view.up());
  fov.set(view.fov());
  ctx->unlock();
}

GuiExtendedView::GuiExtendedView(GuiContext* ctx)
: GuiVar(ctx), eyep(ctx->subVar("eyep")),
  lookat(ctx->subVar("lookat")), up(ctx->subVar("up")),
  fov(ctx->subVar("fov")), eyep_offset(ctx->subVar("eyep_offset")),
  xres(ctx->subVar("xres")), yres(ctx->subVar("yres")), bg(ctx->subVar("bg"))
{
}

GuiExtendedView::~GuiExtendedView()
{
}

ExtendedView
GuiExtendedView::get()
{
  ctx->lock();
  ExtendedView v(eyep.get(), lookat.get(), up.get(), fov.get(), xres.get(),
		 yres.get(), bg.get()*( 1. / 255 ) );
  ctx->unlock();
  return v;
}

void
GuiExtendedView::set(const ExtendedView& view)
{
  ctx->lock();
  eyep.set(view.eyep());
  lookat.set(view.lookat());
  up.set(view.up());
  fov.set(view.fov());
  xres.set(view.xres());
  yres.set(view.yres());
  bg.set( view.bg()*255 );
  ctx->unlock();
}

