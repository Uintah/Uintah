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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>

class GenerateMesh : public Module {
public:
    GenerateMesh(const clString& id);
    GenerateMesh(const GenerateMesh&, int deep);
    virtual ~GenerateMesh();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_GenerateMesh(const clString& id)
{
    return scinew GenerateMesh(id);
}
}

GenerateMesh::GenerateMesh(const clString& id)
: Module("GenerateMesh", id, Filter)
{
    add_iport(scinew SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    add_iport(scinew SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    add_iport(scinew SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    add_iport(scinew SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic));
    // Create the output port
    add_oport(scinew MeshOPort(this, "Geometry", MeshIPort::Atomic));
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
    return scinew GenerateMesh(*this, deep);
}

void GenerateMesh::execute()
{
    NOT_FINISHED("GenerateMesh::execute");
}
