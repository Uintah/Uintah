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

#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Geometry;

class ExtractMesh : public Module {
    ScalarFieldIPort* inports;
    VectorFieldIPort* inportv;
    MeshOPort* outport;
public:
    ExtractMesh(const clString& id);
    virtual ~ExtractMesh();
    virtual void execute();
};

Module* make_ExtractMesh(const clString& id)
{
    return scinew ExtractMesh(id);
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

ExtractMesh::~ExtractMesh()
{
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
} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.1  1999/09/05 01:15:26  dmw
// added all of the old SCIRun mesh modules
//
