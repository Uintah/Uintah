/*
 *  ExtractMeshSF.cc:  Unfinished modules
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
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>

class ExtractMeshSF : public Module {
    ScalarFieldIPort* inport;
    MeshOPort* outport;
public:
    ExtractMeshSF(const clString& id);
    ExtractMeshSF(const ExtractMeshSF&, int deep);
    virtual ~ExtractMeshSF();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ExtractMeshSF(const clString& id)
{
    return scinew ExtractMeshSF(id);
}
}

ExtractMeshSF::ExtractMeshSF(const clString& id)
: Module("ExtractMeshSF", id, Filter)
{
    inport=scinew ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(inport);

    // Create the output port
    outport=scinew MeshOPort(this, "Geometry", MeshIPort::Atomic);
    add_oport(outport);
}

ExtractMeshSF::ExtractMeshSF(const ExtractMeshSF& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("ExtractMeshSF::ExtractMeshSF");
}

ExtractMeshSF::~ExtractMeshSF()
{
}

Module* ExtractMeshSF::clone(int deep)
{
    return scinew ExtractMeshSF(*this, deep);
}

void ExtractMeshSF::execute()
{
    ScalarFieldHandle field;
    if(!inport->get(field))
	return;
    ScalarFieldUG* ugfield=field->getUG();
    if(!ugfield)
	return;
    outport->send(ugfield->mesh);
}
