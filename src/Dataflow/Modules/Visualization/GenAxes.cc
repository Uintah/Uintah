//static char *id="@(#) $Id$";

/*
 *  GenAxes.cc:  
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Mar. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/GeomCone.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECommon/share/share.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

/**************************************
CLASS
   GenAxes
        GenAxes creates an icon in the scene, which displays
        arrows for the X, Y, and Z euclidian directions.

GENERAL INFORMATION

   GenAxes
  
   Author:  David Weinstein<br>
            Department of Computer Science<br>
            University of Utah

   Date:    Mar 1995
   
   C-SAFE
   
   Copyright <C> 1995 SCI Group

KEYWORDS
   Visualization

DESCRIPTION
   GenAxes creates an icon in the scene, which displays
   arrows for the X, Y, and Z euclidian directions.

WARNING
   None

****************************************/

class PSECommonSHARE GenAxes : public Module {
    TCLdouble size;
    int size_changed;
    GeometryOPort* ogeom;
    MaterialHandle dk_red;
    MaterialHandle dk_green;
    MaterialHandle dk_blue;
    MaterialHandle lt_red;
    MaterialHandle lt_green;
    MaterialHandle lt_blue;
public:

        // GROUP:  Constructors:
        ///////////////////////////
        //
        // Constructs an instance of class GenAxes
        //
        // Constructor taking
        //    [in] id as an identifier
        //
    GenAxes(const clString& id);

        // GROUP:  Destructor:
        ///////////////////////////
        // Destructor
    virtual ~GenAxes();


        // GROUP:  Access functions:
        ///////////////////////////
        //
        // execute() - execution scheduled by scheduler
    virtual void execute();


        //////////////////////////
        //
        // tcl_commands - overides tcl_command in base class Module, takes:
        //                          size_changed
    virtual void tcl_command(TCLArgs&, void*);
};

extern "C" PSECommonSHARE Module* make_GenAxes(const clString& id) {
  return new GenAxes(id);
}

//static clString module_name("GenAxes");

GenAxes::GenAxes(const clString& id)
: Module("GenAxes", id, Source), size_changed(1), size("size", id, this)
{
   // Create the output port
   ogeom = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
   add_oport(ogeom);
   dk_red = scinew Material(Color(0,0,0), Color(.2,0,0),
			 Color(.5,.5,.5), 20);
   dk_green = scinew Material(Color(0,0,0), Color(0,.2,0),
			   Color(.5,.5,.5), 20);
   dk_blue = scinew Material(Color(0,0,0), Color(0,0,.2),
			  Color(.5,.5,.5), 20);
   lt_red = scinew Material(Color(0,0,0), Color(.8,0,0),
			 Color(.5,.5,.5), 20);
   lt_green = scinew Material(Color(0,0,0), Color(0,.8,0),
			   Color(.5,.5,.5), 20);
   lt_blue = scinew Material(Color(0,0,0), Color(0,0,.8),
			  Color(.5,.5,.5), 20);
}

GenAxes::~GenAxes()
{
}

void GenAxes::execute()
{
//   if (!size_changed) return;
   size_changed=0;
   ogeom->delAll();
   GeomGroup* xp = scinew GeomGroup;
   GeomGroup* yp = scinew GeomGroup;
   GeomGroup* zp = scinew GeomGroup;
   GeomGroup* xn = scinew GeomGroup;
   GeomGroup* yn = scinew GeomGroup;
   GeomGroup* zn = scinew GeomGroup;
   double sz=size.get();
//   cerr << "Size= " <<size.get() << "\n";
   xp->add(scinew GeomCylinder(Point(0,0,0), Point(sz, 0, 0), sz/20));
   xp->add(scinew GeomCone(Point(sz, 0, 0), Point(sz+sz/5, 0, 0), sz/10, 0));
   yp->add(scinew GeomCylinder(Point(0,0,0), Point(0, sz, 0), sz/20));
   yp->add(scinew GeomCone(Point(0, sz, 0), Point(0, sz+sz/5, 0), sz/10, 0));
   zp->add(scinew GeomCylinder(Point(0,0,0), Point(0, 0, sz), sz/20));
   zp->add(scinew GeomCone(Point(0, 0, sz), Point(0, 0, sz+sz/5), sz/10, 0));
   xn->add(scinew GeomCylinder(Point(0,0,0), Point(-sz, 0, 0), sz/20));
   xn->add(scinew GeomCone(Point(-sz, 0, 0), Point(-sz-sz/5, 0, 0), sz/10, 0));
   yn->add(scinew GeomCylinder(Point(0,0,0), Point(0, -sz, 0), sz/20));
   yn->add(scinew GeomCone(Point(0, -sz, 0), Point(0, -sz-sz/5, 0), sz/10, 0));
   zn->add(scinew GeomCylinder(Point(0,0,0), Point(0, 0, -sz), sz/20));
   zn->add(scinew GeomCone(Point(0, 0, -sz), Point(0, 0, -sz-sz/5), sz/10, 0));
   GeomGroup* all=scinew GeomGroup;
   all->add(scinew GeomMaterial(xp, lt_red));
   all->add(scinew GeomMaterial(yp, lt_green));
   all->add(scinew GeomMaterial(zp, lt_blue));
   all->add(scinew GeomMaterial(xn, dk_red));
   all->add(scinew GeomMaterial(yn, dk_green));
   all->add(scinew GeomMaterial(zn, dk_blue));
   ogeom->addObj(all, "Axes");
   ogeom->flushViews();
}

void GenAxes::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("GenAxes needs a minor command");
	return;
    }
    if (args[1] == "size_changed") {
	size_changed=1;
	want_to_execute();
    } else {
	Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.8  2000/06/07 00:09:25  moulding
// changed the PSECOMMONSHARE macro to PSECommonSHARE to conform to the
// style used by the module maker
//
// Revision 1.7  2000/03/17 09:27:30  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/11/10 19:40:01  moulding
// first commit of an NT'ified module for the new PSE
//
// Revision 1.5  1999/08/25 03:48:06  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:56  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:05  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:48  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:12  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
