//static char *id="@(#) $Id$";

/*
 *  TransformMesh.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;
using namespace SCICore::Containers;
using SCICore::Geometry::Transform;

class TransformMesh : public Module {
    MeshIPort *iport;
    MatrixIPort *imat;
    MeshOPort *oport;
    void MatToTransform(MatrixHandle mH, Transform& t);
public:
    TransformMesh(const clString& id);
    virtual ~TransformMesh();
    virtual void execute();
};

Module* make_TransformMesh(const clString& id) {
  return new TransformMesh(id);
}

TransformMesh::TransformMesh(const clString& id)
: Module("TransformMesh", id, Source)
{
    // Create the input port
    iport = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(iport);
    imat = scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(imat);
    oport = scinew MeshOPort(this, "Mesh", MeshIPort::Atomic);
    add_oport(oport);
}

TransformMesh::~TransformMesh()
{
}

void TransformMesh::MatToTransform(MatrixHandle mH, Transform& t) {
    double a[16];
    double *p=&(a[0]);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            *p++=(*mH.get_rep())[i][j];
    t.set(a);
}

void TransformMesh::execute()
{
    MeshHandle meshIH;
    iport->get(meshIH);
    if (!meshIH.get_rep()) return;

    MatrixHandle mIH;
    imat->get(mIH);
    if (!mIH.get_rep()) return;
    if ((mIH->nrows() != 4) || (mIH->ncols() != 4)) return;
    Transform t;
    MatToTransform(mIH, t);

    meshIH.detach();
    for (int i=0; i<meshIH->nodes.size(); i++) {
	meshIH->nodes[i]->p = t.project(meshIH->nodes[i]->p);
    }

    oport->send(meshIH);
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.1  2000/03/13 05:33:53  dmw
// Transforms are done the same way for ScalarFields, Surfaces and Meshes now - build the transform with the BldTransform module, and then pipe the output matrix into a Transform{Field,Surface,Mesh} module
//
