/*
 *  ExtractMesh.cc:  Unfinished modules
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
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/VectorFieldPort.h>
#include <Datatypes/VectorFieldUG.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>

class ExtractMesh : public Module {
    ScalarFieldIPort* inports;
    VectorFieldIPort* inportv;
    MeshOPort* outport;
public:
    ExtractMesh(const clString& id);
    ExtractMesh(const ExtractMesh&, int deep);
    virtual ~ExtractMesh();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ExtractMesh(const clString& id)
{
    return scinew ExtractMesh(id);
}
}

ExtractMesh::ExtractMesh(const clString& id)
: Module("ExtractMesh", id, Filter)
{
    inports=scinew ScalarFieldIPort(this, "Scalars", ScalarFieldIPort::Atomic);
    add_iport(inports);
    inportv=scinew VectorFieldIPort(this, "Vectors", VectorFieldIPort::Atomic);
    add_iport(inportv);

    // Create the output port
    outport=scinew MeshOPort(this, "Geometry", MeshIPort::Atomic);
    add_oport(outport);
}

ExtractMesh::ExtractMesh(const ExtractMesh& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("ExtractMesh::ExtractMesh");
}

ExtractMesh::~ExtractMesh()
{
}

Module* ExtractMesh::clone(int deep)
{
    return scinew ExtractMesh(*this, deep);
}

void ExtractMesh::execute()
{
    ScalarFieldHandle sf;
    VectorFieldHandle vf;
    if(inports->get(sf)) {
	ScalarFieldUG* ugfield=sf->getUG();
	if(!ugfield)
	    return;
	outport->send(ugfield->mesh);
    } else {
	if (!inportv->get(vf)) return;
	VectorFieldUG* ugfield=vf->getUG();
	if (!ugfield)
	    return;
	outport->send(ugfield->mesh);
    }
}
