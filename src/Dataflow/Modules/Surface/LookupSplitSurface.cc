
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

#include <Core/Containers/String.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Surface.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {


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

    int mycond=mIH->element(ix)->cond;
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

} // End namespace SCIRun

