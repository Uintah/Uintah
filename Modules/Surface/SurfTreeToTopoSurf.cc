
/*
 *  SurfTreeToTopoSurf.cc:  Rescale a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/TopoSurfTree.h>
#include <Geometry/BBox.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

class SurfTreeToTopoSurf : public Module {
    SurfaceIPort* isurface;
    SurfaceOPort* osurface;
    SurfaceHandle osh;
    int generation;
public:
    SurfTreeToTopoSurf(const clString& id);
    SurfTreeToTopoSurf(const SurfTreeToTopoSurf&, int deep);
    virtual ~SurfTreeToTopoSurf();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SurfTreeToTopoSurf(const clString& id)
{
    return new SurfTreeToTopoSurf(id);
}
}

//static clString module_name("SurfTreeToTopoSurf");

SurfTreeToTopoSurf::SurfTreeToTopoSurf(const clString& id)
: Module("SurfTreeToTopoSurf", id, Filter), generation(-1), osh(0)
{
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    // Create the output port
    osurface=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

SurfTreeToTopoSurf::SurfTreeToTopoSurf(const SurfTreeToTopoSurf& copy, int deep)
: Module(copy, deep), generation(-1), osh(0)
{
    NOT_FINISHED("SurfTreeToTopoSurf::SurfTreeToTopoSurf");
}

SurfTreeToTopoSurf::~SurfTreeToTopoSurf()
{
}

Module* SurfTreeToTopoSurf::clone(int deep)
{
    return new SurfTreeToTopoSurf(*this, deep);
}

void SurfTreeToTopoSurf::execute()
{
    SurfaceHandle isurf;
    if(!isurface->get(isurf))
	return;
    if (isurf->generation == generation) { osurface->send(osh); return; }
    generation=isurf->generation;

    SurfTree *st=isurf->getSurfTree();
    if (!st) {
	cerr << "SurfTreeToTopoSurf only works on SurfTrees!\n";
	return;
    }
    TopoSurfTree* topo=st->toTopoSurfTree();
    osh=topo;
    osurface->send(osh);
    return;
}
