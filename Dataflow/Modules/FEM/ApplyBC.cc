
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

#include <Core/Util/NotFinished.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


class ApplyBC : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;
public:
    ApplyBC(const clString& id);
    virtual ~ApplyBC();
    virtual void execute();
};

extern "C" Module* make_ApplyBC(const clString& id) {
  return new ApplyBC(id);
}

ApplyBC::ApplyBC(const clString& id)
: Module("ApplyBC", id, Filter)
{
    iport=scinew SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic);
    add_oport(oport);
}

ApplyBC::~ApplyBC()
{
}

void ApplyBC::execute()
{
    SurfaceHandle surf;
    if(!iport->get(surf))
	return;
    NOT_FINISHED("ApplyBC::execute");
    oport->send(surf);
}

} // End namespace SCIRun

