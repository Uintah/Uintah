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

#include <GenSurface/GenSurface.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <SurfacePort.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

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
