/*
 *  GenerateMesh.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <GenerateMesh/GenerateMesh.h>
#include <MeshPort.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <SurfacePort.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_GenerateMesh()
{
    return new GenerateMesh;
}

static RegisterModule db1("Unfinished", "GenerateMesh", make_GenerateMesh);

GenerateMesh::GenerateMesh()
: Module("GenerateMesh", Filter)
{
    add_iport(new SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    add_iport(new SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    add_iport(new SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    add_iport(new SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    // Create the output port
    add_oport(new MeshOPort(this, "Geometry", MeshIPort::Atomic));
}

GenerateMesh::GenerateMesh(const GenerateMesh& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("GenerateMesh::GenerateMesh");
}

GenerateMesh::~GenerateMesh()
{
}

Module* GenerateMesh::clone(int deep)
{
    return new GenerateMesh(*this, deep);
}

void GenerateMesh::execute()
{
    NOT_FINISHED("GenerateMesh::execute");
}
