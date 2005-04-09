/*
 *  BCNodesToSurf.cc:  Unfinished modules
 *
 *  Written by:
 *   Peter-Pike Sloan and David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geom/Triangles.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>

using sci::MeshHandle;
using sci::Element;

class BCNodesToSurf : public Module {
    MeshIPort* inport;
    SurfaceOPort* osurf;
public:
    BCNodesToSurf(const clString& id);
    BCNodesToSurf(const BCNodesToSurf&, int deep);
    virtual ~BCNodesToSurf();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_BCNodesToSurf(const clString& id)
{
    return scinew BCNodesToSurf(id);
}
};

BCNodesToSurf::BCNodesToSurf(const clString& id)
: Module("BCNodesToSurf", id, Filter)
{
   // Create the input port
    inport=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inport);
    osurf=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurf);
}

BCNodesToSurf::BCNodesToSurf(const BCNodesToSurf& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("BCNodesToSurf::BCNodesToSurf");
}

BCNodesToSurf::~BCNodesToSurf()
{
}

Module* BCNodesToSurf::clone(int deep)
{
    return scinew BCNodesToSurf(*this, deep);
}

void BCNodesToSurf::execute()
{
    MeshHandle mesh;
    if (!inport->get(mesh))
	return;

    TriSurface *ts = new TriSurface;
    
    for (int i=0; i<mesh->nodes.size(); i++) {
	if (mesh->nodes[i]->bc) {
	    ts->points.add(mesh->nodes[i]->p);
	    ts->bcIdx.add(ts->bcIdx.size());
	    ts->bcVal.add(mesh->nodes[i]->bc->value);
	}
    }

    SurfaceHandle sh(ts);
    osurf->send(sh);
}
