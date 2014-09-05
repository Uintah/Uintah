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

#include <Core/GuiInterface/TCLArgs.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Dataflow/Widgets/GuiView.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

GuiView::GuiView(const string& name, const string& id, Part* part)
: GuiVar(name, part), eyep(name+"_eyep", part),
  lookat(name+"_lookat", part), up(name+"_up", part),
  fov(name+"_fov", part), eyep_offset(name+"_eyep_offset", part)
{
}

GuiView::~GuiView()
{
}

void GuiView::reset() {
  eyep.reset();
  lookat.reset();
  up.reset();
  fov.reset();
  eyep_offset.reset();
}

View
GuiView::get()
{
  View v(eyep.get(), lookat.get(), up.get(), fov.get());
  return v;
}

void
GuiView::set(const View& view)
{
  eyep.set(view.eyep());
  lookat.set(view.lookat());
  up.set(view.up());
  fov.set(view.fov());
}


void
GuiView::emit(ostream& out, string& midx)
{
  eyep.emit(out, midx);
  lookat.emit(out, midx);
  up.emit(out, midx);
  fov.emit(out, midx);
}



GuiExtendedView::GuiExtendedView( const string& name, const string& id,
				  Part* part )
  : GuiVar(name, part), eyep(name+"_eyep", part),
    lookat(name+"_lookat", part), up(name+"_up", part),
    fov(name+"_fov", part), eyep_offset(name+"_eyep_offset", part),
    xres(name+"_xres", part), yres(name+"_yres", part), 
    bg(name+"_bg", part)
{
}

GuiExtendedView::~GuiExtendedView()
{
}


void GuiExtendedView::reset() {
  eyep.reset();
  lookat.reset();
  up.reset();
  fov.reset();
  eyep_offset.reset();
  xres.reset();
  yres.reset();
  bg.reset();
}

ExtendedView
GuiExtendedView::get()
{
  ExtendedView v(eyep.get(), lookat.get(), up.get(), fov.get(), xres.get(),
		 yres.get(), bg.get()*( 1. / 255 ) );
  return v;
}

void
GuiExtendedView::set(const ExtendedView& view)
{
  eyep.set(view.eyep());
  lookat.set(view.lookat());
  up.set(view.up());
  fov.set(view.fov());
  xres.set(view.xres());
  yres.set(view.yres());
  bg.set( view.bg()*255 );
}


void
GuiExtendedView::emit(ostream& out, string& midx)
{
  eyep.emit(out, midx);
  lookat.emit(out, midx);
  up.emit(out, midx);
  fov.emit(out, midx);
  xres.emit(out, midx);
  yres.emit(out, midx);
  bg.emit(out, midx);
}

} // End namespace SCIRun

