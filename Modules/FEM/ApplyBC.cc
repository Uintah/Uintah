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

class ApplyBC : public Module {
public:
    ApplyBC(const clString& id);
    ApplyBC(const ApplyBC&, int deep);
    virtual ~ApplyBC();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_ApplyBC(const clString& id)
{
    return new ApplyBC(id);
}

static RegisterModule db1("Unfinished", "ApplyBC", make_ApplyBC);

ApplyBC::ApplyBC(const clString& id)
: Module("ApplyBC", id, Filter)
{
    add_iport(new SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    // Create the output port
    add_oport(new SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic));
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
    return new ApplyBC(*this, deep);
}

void ApplyBC::execute()
{
    NOT_FINISHED("ApplyBC::execute");
}
