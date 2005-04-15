//static char *id="@(#) $Id$";

/*
 *  LookupSplitSurface.cc:  Lookup data values for a surface in a scalarfield
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Datatypes/Surface.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
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
using namespace SCICore::TclInterface;
using SCICore::Geometry::AffineCombination;

class LookupSplitSurface : public Module {
  SurfaceIPort *iport;
  MeshIPort *imesh;
  ColumnMatrixIPort *icond;
  SurfaceOPort *oport;
  TCLstring splitDirTCL;
  TCLdouble splitValTCL;
public:
  LookupSplitSurface(const clString& id);
  virtual ~LookupSplitSurface();
  virtual void execute();
};

extern "C" Module* make_LookupSplitSurface(const clString& id) {
  return new LookupSplitSurface(id);
}

LookupSplitSurface::LookupSplitSurface(const clString& id)
  : Module("LookupSplitSurface", id, Source), 
  splitDirTCL("splitDirTCL", id, this),
  splitValTCL("splitValTCL", id, this)
{
  // Create the input port
  iport = scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
  add_iport(iport);
  imesh = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
  add_iport(imesh);
  icond = scinew ColumnMatrixIPort(this, "Conductivities", 
				   ColumnMatrixIPort::Atomic);
  add_iport(icond);
  oport = scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
  add_oport(oport);
}

LookupSplitSurface::~LookupSplitSurface()
{
}

void LookupSplitSurface::execute()
{
  SurfaceHandle sIH;
  iport->get(sIH);
  if (!sIH.get_rep()) return;

  MeshHandle mIH;
  imesh->get(mIH);
  if (!mIH.get_rep()) return;

  ColumnMatrixHandle cIH;
  icond->get(cIH);
  if (!cIH.get_rep()) return;
  if (cIH->nrows() != (mIH->cond_tensors.size()*2)) {
    cerr << "Error - different number of conductivities!\n";
    return;
  }

  if (!dynamic_cast<TriSurface*>(sIH.get_rep())) {
    cerr << "Error -- LookupSplitSurface only knows how to deal w/ trisurfaces.\n";
    return;
  }

  reset_vars();
  clString splitDir=splitDirTCL.get();
  double splitVal=splitValTCL.get();

  TriSurface *ts=new TriSurface(*(dynamic_cast<TriSurface*>(sIH.get_rep())));
  ts->bcVal.resize(0);
  ts->bcIdx.resize(0);
  int i;
  int ix=0;
  ColumnMatrix *c=cIH.get_rep();
  int half=c->nrows()/2;
  ts->valType=TriSurface::FaceType;
  for (i=0; i<ts->elements.size(); i++) {
    ts->bcIdx.add(i);
    TSElement *e=ts->elements[i];
    Point p(AffineCombination(ts->points[e->i1], 1./3.,
			      ts->points[e->i2], 1./3.,
			      ts->points[e->i3], 1./3.));
    if (!mIH->locate(p, ix, 1.e-4, 1.e-4)) {
      cerr << "LookupSplitSurface -- point out of bounds.\n";
      return;
    }

    int mycond=mIH->elems[ix]->cond;
    if ((splitDir == "X") && p.x()<splitVal)
      mycond+=half;
    else if ((splitDir == "Y") && p.y()<splitVal)
      mycond+=half;
    else if ((splitDir == "Z") && p.z()<splitVal)
      mycond+=half;

    ts->bcVal.add(log((*c)[mycond]));
  }
  oport->send(ts);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1.2.1  2000/10/31 02:24:34  dmw
// Merging PSECommon changes from HEAD to FIELD_REDESIGN branch
//
// Revision 1.1  2000/10/29 04:34:56  dmw
// BuildFEMatrix -- ground an arbitrary node
// SolveMatrix -- when preconditioning, be careful with 0's on diagonal
// MeshReader -- build the grid when reading
// SurfToGeom -- support node normals
// IsoSurface -- fixed tet mesh bug
// MatrixWriter -- support split file (header + raw data)
//
// LookupSplitSurface -- split a surface across a place and lookup values
// LookupSurface -- find surface nodes in a sfug and copy values
// Current -- compute the current of a potential field (- grad sigma phi)
// LocalMinMax -- look find local min max points in a scalar field
//
