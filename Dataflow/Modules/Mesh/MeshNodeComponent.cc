/*
 *  MeshNodeComponent.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

namespace SCIRun {


class MeshNodeComponent : public Module {
    MeshIPort* iport;
    ColumnMatrixOPort* oport;
    TCLstring compTCL;
public:
    MeshNodeComponent(const clString& id);
    virtual ~MeshNodeComponent();
    virtual void execute();
};

extern "C" Module* make_MeshNodeComponent(const clString& id)
{
    return scinew MeshNodeComponent(id);
}

MeshNodeComponent::MeshNodeComponent(const clString& id)
: Module("MeshNodeComponent", id, Filter), compTCL("compTCL", id, this)
{
   // Create the input port
    iport=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(iport);
    oport=scinew ColumnMatrixOPort(this, "Component", ColumnMatrixIPort::Atomic);
    add_oport(oport);
}

MeshNodeComponent::~MeshNodeComponent()
{
}

void MeshNodeComponent::execute()
{
    MeshHandle mesh;
    if (!iport->get(mesh))
	return;

    ColumnMatrix *comp = new ColumnMatrix(mesh->nodes.size());
    int i;
    if (compTCL.get() == "x") {
	for (i=0; i<mesh->nodes.size(); i++)
	    (*comp)[i]=mesh->nodes[i]->p.x();
    } else if (compTCL.get() == "y") {
	for (i=0; i<mesh->nodes.size(); i++)
	    (*comp)[i]=mesh->nodes[i]->p.y();
    } else { // if (compTCL.get() == "z") {
	for (i=0; i<mesh->nodes.size(); i++)
	    (*comp)[i]=mesh->nodes[i]->p.z();
    }

    oport->send(ColumnMatrixHandle(comp));
}

} // End namespace SCIRun


