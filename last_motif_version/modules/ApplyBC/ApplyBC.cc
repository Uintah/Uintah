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

#include <ApplyBC/ApplyBC.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <SurfacePort.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_ApplyBC()
{
    return new ApplyBC;
}

static RegisterModule db1("Unfinished", "ApplyBC", make_ApplyBC);

ApplyBC::ApplyBC()
: UserModule("ApplyBC", Filter)
{
    add_iport(new SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    // Create the output port
    add_oport(new SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic));
}

ApplyBC::ApplyBC(const ApplyBC& copy, int deep)
: UserModule(copy, deep)
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
