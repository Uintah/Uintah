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


#include <TCL/TCL.h>
#include <TCL/TCLTask.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

#include <tcl/tcl/tcl.h>
#include <Geom/TCLView.h>



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
