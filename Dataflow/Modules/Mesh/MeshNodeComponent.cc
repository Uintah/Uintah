/*
 *  MeshNodeCore/CCA/Component.cc:  Unfinished modules
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


class MeshNodeCore/CCA/Component : public Module {
    MeshIPort* iport;
    ColumnMatrixOPort* oport;
    TCLstring compTCL;
public:
    MeshNodeCore/CCA/Component(const clString& id);
    virtual ~MeshNodeCore/CCA/Component();
    virtual void execute();
};

extern "C" Module* make_MeshNodeCore/CCA/Component(const clString& id)
{
    return scinew MeshNodeCore/CCA/Component(id);
}

MeshNodeCore/CCA/Component::MeshNodeCore/CCA/Component(const clString& id)
: Module("MeshNodeCore/CCA/Component", id, Filter), compTCL("compTCL", id, this)
{
   // Create the input port
    iport=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(iport);
    oport=scinew ColumnMatrixOPort(this, "Core/CCA/Component", ColumnMatrixIPort::Atomic);
    add_oport(oport);
}

MeshNodeCore/CCA/Component::~MeshNodeCore/CCA/Component()
{
}

void MeshNodeCore/CCA/Component::execute()
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


