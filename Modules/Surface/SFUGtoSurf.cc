
/*
 *  SFUGtoSurf.cc:  Dumb module -- blow away elements and make nodes into
 *			a trisurface with no triangles
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

class SFUGtoSurf : public Module {
    ScalarFieldIPort* isf;
    SurfaceOPort* osurf;
public:
    SFUGtoSurf(const clString& id);
    SFUGtoSurf(const SFUGtoSurf&, int deep);
    virtual ~SFUGtoSurf();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SFUGtoSurf(const clString& id)
{
    return new SFUGtoSurf(id);
}
};

static clString module_name("SFUGtoSurf");

SFUGtoSurf::SFUGtoSurf(const clString& id)
: Module("SFUGtoSurf", id, Filter)
{
    isf=scinew ScalarFieldIPort(this, "SF", ScalarFieldIPort::Atomic);
    add_iport(isf);
    // Create the output port
    osurf=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurf);
}

SFUGtoSurf::SFUGtoSurf(const SFUGtoSurf& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SFUGtoSurf::SFUGtoSurf");
}

SFUGtoSurf::~SFUGtoSurf()
{
}

Module* SFUGtoSurf::clone(int deep)
{
    return new SFUGtoSurf(*this, deep);
}

void SFUGtoSurf::execute() {
    ScalarFieldHandle isfh;
    if(!isf->get(isfh))
	return;
    ScalarFieldUG* sfug=isfh->getUG();
    if (!sfug) return;
    TriSurface *ts=scinew TriSurface;
    int npts=sfug->mesh->nodes.size();
    ts->points.resize(npts);
    for (int i=0; i<npts; i++) {
	ts->points[i]=sfug->mesh->nodes[i]->p;
	if (sfug->mesh->nodes[i]->bc) {
	    ts->bcIdx.add(i);
	    ts->bcVal.add(sfug->mesh->nodes[i]->bc->value);
	}
    }
    SurfaceHandle osh(ts);
    osurf->send(osh);
    return;
}
