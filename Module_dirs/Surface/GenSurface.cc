/*
 *  GenSurface.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>

class GenSurface : public Module {
    TCLstring surfacetype;
    TCLPoint cyl_p1;
    TCLPoint cyl_p2;
    TCLdouble cyl_rad;
    TCLint cyl_nu;
    TCLint cyl_nv;
    TCLint cyl_ndiscu;

    SurfaceOPort* outport;
public:
    GenSurface(const clString& id);
    GenSurface(const GenSurface&, int deep);
    virtual ~GenSurface();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_GenSurface(const clString& id)
{
    return new GenSurface(id);
}

static RegisterModule db1("Unfinished", "GenSurface", make_GenSurface);

GenSurface::GenSurface(const clString& id)
: Module("GenSurface", id, Source), surfacetype("surfacetype", id, this),
  cyl_p1("cyl_p1", id, this), cyl_p2("cyl_p2", id, this),
  cyl_rad("cyl_rad", id, this), cyl_nu("cyl_nu", id, this),
  cyl_nv("cyl_nv", id, this), cyl_ndiscu("cyl_ndiscu", id, this)
{
    // Create the output port
    outport=new SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic);
    add_oport(outport);
    surfacetype.set("cylinder");
}

GenSurface::GenSurface(const GenSurface& copy, int deep)
: Module(copy, deep), surfacetype("surfacetype", id, this),
  cyl_p1("cyl_p1", id, this), cyl_p2("cyl_p2", id, this),
  cyl_rad("cyl_rad", id, this), cyl_nu("cyl_nu", id, this),
  cyl_nv("cyl_nv", id, this), cyl_ndiscu("cyl_ndiscu", id, this)
{
    NOT_FINISHED("GenSurface::GenSurface");
}

GenSurface::~GenSurface()
{
}

Module* GenSurface::clone(int deep)
{
    return new GenSurface(*this, deep);
}

void GenSurface::execute()
{
    Surface* surf=0;
    clString st(surfacetype.get());
    if(st=="cylinder"){
	surf=new CylinderSurface(cyl_p1.get(), cyl_p2.get(), cyl_rad.get(),
				 cyl_nu.get(), cyl_nv.get(), cyl_ndiscu.get());
    } else {
	error("Unknown surfacetype: "+st);
    }
    if(surf)
        outport->send(SurfaceHandle(surf));
}
