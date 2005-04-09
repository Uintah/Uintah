
/*
 *  SurfExtractBCNodes.cc:  Extract the nodes with BC's
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
#include <Datatypes/SurfacePort.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/ScalarTriSurface.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Math/Expon.h>
#include <Math/MusilRNG.h>
#include <Math/Trig.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

class SurfExtractBCNodes : public Module {
    SurfaceIPort* isurface;
    SurfaceOPort* osurface;
public:
    SurfExtractBCNodes(const clString& id);
    SurfExtractBCNodes(const SurfExtractBCNodes&, int deep);
    virtual ~SurfExtractBCNodes();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SurfExtractBCNodes(const clString& id)
{
    return new SurfExtractBCNodes(id);
}
}

static clString module_name("SurfExtractBCNodes");

SurfExtractBCNodes::SurfExtractBCNodes(const clString& id)
: Module("SurfExtractBCNodes", id, Filter)
{
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    // Create the output port
    osurface=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

SurfExtractBCNodes::SurfExtractBCNodes(const SurfExtractBCNodes& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SurfExtractBCNodes::SurfExtractBCNodes");
}

SurfExtractBCNodes::~SurfExtractBCNodes()
{
}

Module* SurfExtractBCNodes::clone(int deep)
{
    return new SurfExtractBCNodes(*this, deep);
}

void SurfExtractBCNodes::execute()
{
    SurfaceHandle isurf;
    if(!isurface->get(isurf))
	return;
    TriSurface *ts=isurf->getTriSurface();
    if (!ts) {
	cerr << "SurfExtractBCNodes: error -- need trisurface.\n";
	return;
    }

    TriSurface *ts2 = new TriSurface;
    ts2->points.resize(ts->bcIdx.size());
    ts2->bcIdx.resize(ts->bcIdx.size());
    ts2->bcVal.resize(ts->bcIdx.size());

    for (int i=0; i<ts->bcIdx.size(); i++) {
	int idx=ts->bcIdx[i];
	ts2->points[i]=ts->points[idx];
	ts2->bcIdx[i]=i;
	ts2->bcVal[i]=ts->bcVal[i];
    }

    if (ts2->normType == TriSurface::PointType && 
	ts2->normals.size() == ts2->bcIdx.size()) {
	ts->normals=ts2->normals;
	ts->normType = TriSurface::PointType;
    }
    SurfaceHandle osh(ts2);
    osurface->send(osh);
    return;
}
