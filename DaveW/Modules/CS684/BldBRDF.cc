//static char *id="@(#) $Id$";

/*
 *  BldBRDF.cc:  Take in a scene (through a VoidStarPort), and output the scene
 *		   Geometry and a rendering window
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/Tester/RigorousTest.h>
#include <SCICore/Util/NotFinished.h>

#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::CoreDatatypes;
using namespace SCICore::TclInterface;

class BldBRDF : public Module {
    SurfaceIPort *iSurf;
    SurfaceOPort *oSurf;

    TCLstring theta_expr;
    TCLstring phi_expr;
    int tcl_exec;
    int init;
public:
    BldBRDF(const clString& id);
    virtual ~BldBRDF();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
};

Module* make_BldBRDF(const clString& id)
{
    return scinew BldBRDF(id);
}

static clString module_name("BldBRDF");

BldBRDF::BldBRDF(const clString& id)
: Module("BldBRDF", id, Filter), theta_expr("theta_expr", id, this),
  phi_expr("phi_expr", id, this), tcl_exec(0), init(0)
{
    // Create the input port
    iSurf = scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(iSurf);
    // Create the output port
    oSurf = scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(oSurf);
    init=0;
}

BldBRDF::~BldBRDF()
{
}

void BldBRDF::execute()
{
    return;
}
void BldBRDF::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "tcl_exec") {
	tcl_exec=1;
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}
} // End namespace Modules
} // End namespace DaveW
//
// $Log$
// Revision 1.1  1999/08/24 06:22:59  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:10  dmw
// Added and updated DaveW Datatypes/Modules
//
//
