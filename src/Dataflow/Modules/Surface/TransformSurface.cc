/*
 *  TransformSurface.cc:  Rotate and flip field to get it into "standard" view
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
#include <Core/Datatypes/Surface.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {


class TransformSurface : public Module {
    SurfaceIPort *iport;
    MatrixIPort *imat;
    SurfaceOPort *oport;
    void MatToTransform(MatrixHandle mH, Transform& t);
public:
    TransformSurface(const clString& id);
    virtual ~TransformSurface();
    virtual void execute();
};

extern "C" Module* make_TransformSurface(const clString& id) {
  return new TransformSurface(id);
}

TransformSurface::TransformSurface(const clString& id)
: Module("TransformSurface", id, Source)
{
    // Create the input port
    iport = scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(iport);
    imat = scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(imat);
    oport = scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(oport);
}

TransformSurface::~TransformSurface()
{
}

void TransformSurface::MatToTransform(MatrixHandle mH, Transform& t) {
    double a[16];
    double *p=&(a[0]);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            *p++=(*mH.get_rep())[i][j];
    t.set(a);
}

void TransformSurface::execute()
{
    SurfaceHandle sIH;
    iport->get(sIH);
    if (!sIH.get_rep()) return;

    MatrixHandle mIH;
    imat->get(mIH);
    if (!mIH.get_rep()) return;
    if ((mIH->nrows() != 4) || (mIH->ncols() != 4)) return;
    Transform t;
    MatToTransform(mIH, t);

    Array1<NodeHandle> nodes;

    SurfaceHandle sss=sIH;

    sIH.detach();
    sIH->get_surfnodes(nodes);
    for (int i=0; i<nodes.size(); i++) {
	nodes[i]->p = t.project(nodes[i]->p);
    }
    sIH->set_surfnodes(nodes);

    oport->send(sIH);
}

} // End namespace SCIRun
