/*
 *  Gradient.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Gradient/Gradient.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <ScalarFieldPort.h>
#include <SurfacePort.h>
#include <VectorFieldPort.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_Gradient(const clString& id)
{
    return new Gradient(id);
}

static RegisterModule db1("Unfinished", "Gradient", make_Gradient);

Gradient::Gradient(const clString& id)
: Module("Gradient", id, Filter)
{
    add_iport(new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic));
    // Create the output port
    add_oport(new VectorFieldOPort(this, "Geometry", VectorFieldIPort::Atomic));
}

Gradient::Gradient(const Gradient& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Gradient::Gradient");
}

Gradient::~Gradient()
{
}

Module* Gradient::clone(int deep)
{
    return new Gradient(*this, deep);
}

void Gradient::execute()
{
    NOT_FINISHED("Gradient::execute");
}
