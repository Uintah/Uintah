/*
 *  ApplyBC.cc:  Unfinished modules
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
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>

class ApplyBC : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;
public:
    ApplyBC(const clString& id);
    ApplyBC(const ApplyBC&, int deep);
    virtual ~ApplyBC();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_ApplyBC(const clString& id)
{
    return scinew ApplyBC(id);
}

static RegisterModule db1("Unfinished", "ApplyBC", make_ApplyBC);

ApplyBC::ApplyBC(const clString& id)
: Module("ApplyBC", id, Filter)
{
    iport=scinew SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic);
    add_oport(oport);
}

ApplyBC::ApplyBC(const ApplyBC& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("ApplyBC::ApplyBC");
}

ApplyBC::~ApplyBC()
{
}

Module* ApplyBC::clone(int deep)
{
    return scinew ApplyBC(*this, deep);
}

void ApplyBC::execute()
{
    SurfaceHandle surf;
    if(!iport->get(surf))
	return;
    NOT_FINISHED("ApplyBC::execute");
    oport->send(surf);
}
