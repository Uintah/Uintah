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

#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geom/Color.h>
#include <Geom/GeomOpenGL.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Triangles.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>
#include <TCL/TCL.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>
#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

class BldBRDF : public Module {
    SurfaceIPort *iSurf;
    SurfaceOPort *oSurf;

    TCLstring theta_expr;
    TCLstring phi_expr;
    int tcl_exec;
    int init;
public:
    BldBRDF(const clString& id);
    BldBRDF(const BldBRDF&, int deep);
    virtual ~BldBRDF();
    virtual Module* clone(int deep);
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
};

extern "C" {
Module* make_BldBRDF(const clString& id)
{
    return scinew BldBRDF(id);
}
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

BldBRDF::BldBRDF(const BldBRDF& copy, int deep)
: Module(copy, deep), theta_expr("theta_expr", id, this),
  phi_expr("phi_expr", id, this), tcl_exec(0), init(0)
{
    NOT_FINISHED("BldBRDF::BldBRDF");
}

BldBRDF::~BldBRDF()
{
}

Module* BldBRDF::clone(int deep)
{
    return scinew BldBRDF(*this, deep);
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
