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

