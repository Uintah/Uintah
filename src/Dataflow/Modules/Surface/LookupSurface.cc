
/*
 *  LookupSurface.cc:  Lookup data values for a surface in a scalarfield
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
#include <Core/Datatypes/ScalarField.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
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


class LookupSurface : public Module {
  SurfaceIPort *iport;
  ScalarFieldIPort *ifield;
  SurfaceOPort *oport;
  TCLint constElemsTCL;
public:
  LookupSurface(const clString& id);
  virtual ~LookupSurface();
  virtual void execute();
};

extern "C" Module* make_LookupSurface(const clString& id) {
  return new LookupSurface(id);
}

LookupSurface::LookupSurface(const clString& id)
  : Module("LookupSurface", id, Source), 
  constElemsTCL("constElemsTCL", id, this)
{
  // Create the input port
  iport = scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
  add_iport(iport);
  ifield = scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
  add_iport(ifield);
  oport = scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
  add_oport(oport);
}

LookupSurface::~LookupSurface()
{
}

void LookupSurface::execute()
{
  SurfaceHandle sIH;
  iport->get(sIH);
  if (!sIH.get_rep()) return;

  ScalarFieldHandle fIH;
  ifield->get(fIH);
  if (!fIH.get_rep()) return;
  
  if (!dynamic_cast<TriSurface*>(sIH.get_rep())) {
    cerr << "Error -- LookupSurface only knows how to deal w/ trisurfaces.\n";
    return;
  }

  sIH.detach();
  TriSurface *ts=dynamic_cast<TriSurface*>(sIH.get_rep());
  ts->bcVal.resize(0);
  ts->bcIdx.resize(0);
  int i;
  int ix=0;
  double interp;
  reset_vars();
  int constElems=constElemsTCL.get();
  if (constElems) {
    ts->valType=TriSurface::FaceType;
    ts->bcIdx.resize(ts->elements.size());
    for (i=0; i<ts->elements.size(); i++) {
      ts->bcIdx.add(i);
      TSElement *e=ts->elements[i];
      Point p(AffineCombination(ts->points[e->i1], 1./3.,
				ts->points[e->i2], 1./3.,
				ts->points[e->i3], 1./3.));
      if (!fIH->interpolate(p, interp, ix, 1.e-4, 1.e-4)) interp=-1000;
      ts->bcVal.add(interp);
    }
  } else {
    ts->valType=TriSurface::NodeType;
    ts->bcIdx.resize(ts->points.size());
    for (i=0; i<ts->points.size(); i++) {
      ts->bcIdx.add(i);
      
      if (!fIH->interpolate(ts->points[i], interp, ix, 1.e-4, 1.e-4))
	interp=-1000;
      ts->bcVal.add(interp);
    }
  }

  oport->send(sIH);
}

} // End namespace SCIRun

