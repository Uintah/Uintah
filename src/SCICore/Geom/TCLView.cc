/*
 *  TCLView.cc  Structure that provides for easy access of view information.
 *              The view information is interactively provided by the user.
 *
 *  Written by:
 *   Steven Parker
 *   Department of Computer Science
 *   University of Utah
 *
 *   separated from the Salmon code by me (Aleksandra)
 *   in May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <tcl.h>

#include <SCICore/TclInterface/TCL.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geom/TCLView.h>

namespace SCICore {
namespace GeomSpace {

TCLView::TCLView(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), eyep("eyep", str(), tcl),
  lookat("lookat", str(), tcl), up("up", str(), tcl),
  fov("fov", str(), tcl), eyep_offset("eyep_offset", str(), tcl)
{
}

TCLView::~TCLView()
{
}

View
TCLView::get()
{
    TCLTask::lock();
    View v(eyep.get(), lookat.get(), up.get(), fov.get());
    TCLTask::unlock();
    return v;
}

void
TCLView::set(const View& view)
{
    TCLTask::lock();
    eyep.set(view.eyep());
    lookat.set(view.lookat());
    up.set(view.up());
    fov.set(view.fov());
    TCLTask::unlock();
}


void
TCLView::emit(ostream& out)
{
    eyep.emit(out);
    lookat.emit(out);
    up.emit(out);
    fov.emit(out);
}



TCLExtendedView::TCLExtendedView( const clString& name, const clString& id,
				 TCL* tcl )
: TCLvar(name, id, tcl), eyep("eyep", str(), tcl),
  lookat("lookat", str(), tcl), up("up", str(), tcl),
  fov("fov", str(), tcl), eyep_offset("eyep_offset", str(), tcl),
  bg("bg", str(), tcl), xres("xres", str(), tcl), yres("yres", str(), tcl)
{
}

TCLExtendedView::~TCLExtendedView()
{
}


ExtendedView
TCLExtendedView::get()
{
    TCLTask::lock();
    ExtendedView v(eyep.get(), lookat.get(), up.get(), fov.get(), xres.get(),
		   yres.get(), bg.get()*( 1. / 255 ) );
    TCLTask::unlock();
    return v;
}

void
TCLExtendedView::set(const ExtendedView& view)
{
    TCLTask::lock();
    eyep.set(view.eyep());
    lookat.set(view.lookat());
    up.set(view.up());
    fov.set(view.fov());
    xres.set(view.xres());
    yres.set(view.yres());
    bg.set( view.bg()*255 );
    TCLTask::unlock();
}


void
TCLExtendedView::emit(ostream& out)
{
    eyep.emit(out);
    lookat.emit(out);
    up.emit(out);
    fov.emit(out);
    xres.emit(out);
    yres.emit(out);
    bg.emit(out);
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:24  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:52  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//
