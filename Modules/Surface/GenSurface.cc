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
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>

class GenSurface : public Module {
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
: Module("GenSurface", id, Source)
{
    // Create the output port
    add_oport(new SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic));
}

GenSurface::GenSurface(const GenSurface& copy, int deep)
: Module(copy, deep)
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
    NOT_FINISHED("GenSurface::execute");
}
