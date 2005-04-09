/*
 *  TCLView.h   Structure that provides for easy access of view information.
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

#ifndef SCI_project_TCLView_h
#define SCI_project_TCLView_h 1

#include <Geom/View.h>
#include <Geom/TCLGeom.h>
#include <Classlib/String.h>
#include <TCL/TCLvar.h>

class TCLView : public TCLvar {
    TCLPoint eyep;
    TCLPoint lookat;
    TCLVector up;
    TCLdouble fov;
    TCLVector eyep_offset;
public:
    TCLView(const clString& name, const clString& id, TCL* tcl);
    ~TCLView();
    TCLView(const TCLView&);

    View get();
    void set(const View&);
    virtual void emit(ostream& out);
};

class TCLExtendedView : public TCLvar {
    TCLPoint eyep;
    TCLPoint lookat;
    TCLVector up;
    TCLdouble fov;
    TCLVector eyep_offset;

    TCLint   xres;
    TCLint   yres;

public:
    TCLColor bg;
    TCLExtendedView(const clString& name, const clString& id, TCL* tcl);
    ~TCLExtendedView();
    TCLExtendedView(const TCLExtendedView&);

    ExtendedView get();
    void set(const ExtendedView&);
    virtual void emit(ostream& out);
  };

#endif
