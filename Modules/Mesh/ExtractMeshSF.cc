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

#include <ExtractMeshSF/ExtractMeshSF.h>
#include <MeshPort.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <ScalarFieldPort.h>
#include <ScalarFieldUG.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_ExtractMeshSF(const clString& id)
{
    return new ExtractMeshSF(id);
}

static RegisterModule db1("Unfinished", "ExtractMeshSF", make_ExtractMeshSF);

ExtractMeshSF::ExtractMeshSF(const clString& id)
: Module("ExtractMeshSF", id, Filter)
{
    inport=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(inport);

    // Create the output port
    outport=new MeshOPort(this, "Geometry", MeshIPort::Atomic);
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
    return new ExtractMeshSF(*this, deep);
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
