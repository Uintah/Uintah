
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

#include <Core/Containers/String.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {


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

extern "C" Module* make_TransformMesh(const clString& id) {
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

    MeshHandle mmm=meshIH;
    meshIH.detach();
    for (int i=0; i<meshIH->nodesize(); i++)
    {
      meshIH->nodes[i]->p = t.project(meshIH->point(i));
    }
    oport->send(meshIH);
}

} // End namespace SCIRun

