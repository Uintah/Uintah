//static char *id="@(#) $Id$";

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

#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/Surface.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;
using namespace SCICore::Containers;
using SCICore::Geometry::Transform;

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
    sIH.detach();
    sIH->get_surfnodes(nodes);
    for (int i=0; i<nodes.size(); i++) {
	nodes[i]->p = t.project(nodes[i]->p);
    }
    sIH->set_surfnodes(nodes);

    oport->send(sIH);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  2000/03/17 09:27:23  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.1  2000/03/13 05:33:23  dmw
// Transforms are done the same way for ScalarFields, Surfaces and Meshes now - build the transform with the BldTransform module, and then pipe the output matrix into a Transform{Field,Surface,Mesh} module
//
