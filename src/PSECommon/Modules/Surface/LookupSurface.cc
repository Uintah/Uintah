//static char *id="@(#) $Id$";

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

#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
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
