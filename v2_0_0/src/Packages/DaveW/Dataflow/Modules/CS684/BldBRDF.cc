
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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/TriSurfFieldace.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Tester/RigorousTest.h>

namespace DaveW {
using namespace SCIRun;

class BldBRDF : public Module {
    SurfaceIPort *iSurf;
    SurfaceOPort *oSurf;

    GuiString theta_expr;
    GuiString phi_expr;
    int tcl_exec;
    int init;
public:
    BldBRDF(const clString& id);
    virtual ~BldBRDF();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
};

extern "C" Module* make_BldBRDF(const clString& id)
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
} // End namespace DaveW
}
