
/*
 *  LocateNbrhd.cc:  Rescale a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {


class LocateNbrhd : public Module
{
  MeshIPort* imesh_;
  SurfaceIPort* isurf_;

  MatrixOPort* omatrix_;
  MatrixHandle omatrixH_;

  MatrixOPort* omat_;
  MatrixHandle omatH_;

  SurfaceOPort *osurf_;
  SurfaceHandle osurfH_;

  TCLstring method_;
  TCLint zeroTCL_;
  TCLint potMatTCL_;

  int mesh_generation_;
  int surf_generation_;

public:

  LocateNbrhd(const clString& id);
  virtual ~LocateNbrhd();
  virtual void execute();
};


extern "C" Module* make_LocateNbrhd(const clString& id)
{
  return new LocateNbrhd(id);
}


LocateNbrhd::LocateNbrhd(const clString& id)
  : Module("LocateNbrhd", id, Filter),
    method_("method", id, this),
    zeroTCL_("zeroTCL", id, this),
    potMatTCL_("potMatTCL", id, this)
{
  imesh_ = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
  add_iport(imesh_);

  isurf_ = scinew SurfaceIPort(this, "Surface2", SurfaceIPort::Atomic);
  add_iport(isurf_);

  // Create the output ports.
  omat_ = scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
  add_oport(omat_);

  omatrix_ = scinew MatrixOPort(this, "Map", MatrixIPort::Atomic);
  add_oport(omatrix_);

  osurf_ = scinew SurfaceOPort(this, "NearestNodes", SurfaceIPort::Atomic);
  add_oport(osurf_);

  mesh_generation_ = -1;
  surf_generation_ = -1;
}


LocateNbrhd::~LocateNbrhd()
{
}


void
LocateNbrhd::execute()
{
  MeshHandle meshH;
  if (!imesh_->get(meshH))
    return;

  SurfaceHandle surfH;
  if (!isurf_->get(surfH))
    return;

  // TODO: Check gui state also.
  if (mesh_generation_ == meshH->generation &&
      surf_generation_ == surfH->generation &&
      omatrixH_.get_rep())
  {
    omatrix_->send(omatrixH_);
    omat_->send(omatH_);
    osurf_->send(osurfH_);
    return;
  }

  mesh_generation_ = meshH->generation;
  surf_generation_ = surfH->generation;
  clString m(method_.get());

#if 0
  TriSurface *ots = new TriSurface;
  osurfH_ = ots;
  if (m == "project")
  {	
    TriSurface *ts = scinew TriSurface;
	
    // First, set up the data point locations and values in an array.
    Array1<Point> p;

    // Get the right trisurface and grab those vals.
    if (!(ts = surfH->getTriSurface()))
    {

      cerr << "Error - need a trisurface!\n";
      return;
    }
    if (ts->bcIdx.size()==0)
    {
      ts->bcIdx.resize(ts->points.size());
      ts->bcVal.resize(ts->points.size());
      for (int ii = 0; ii<ts->points.size(); ii++)
      {
	ts->bcIdx[ii] = ii;
	ts->bcVal[ii] = 0;
      }
    }
    int i;
    for (i = 0; i<ts->bcIdx.size(); i++)
    {
      p.add(ts->points[ts->bcIdx[i]]);
    }
    ColumnMatrix *mapping = scinew ColumnMatrix(ts->bcIdx.size());
    omatrixH_ = mapping;
    int *rows;
    int *cols;
    double *a;
    SparseRowMatrix *mm;
    if (potMatTCL_.get())
    {
      rows = new int[ts->bcIdx.size()];
      cols = new int[(ts->bcIdx.size()-1)];
      a = new double[(ts->bcIdx.size()-1)];
      mm = scinew SparseRowMatrix(ts->bcIdx.size()-1,
				  meshH->nodesize(),
				  rows, cols,
				  ts->bcIdx.size()-1, a);
      for (i = 0; i<ts->bcIdx.size(); i++)
      { rows[i] = i;
      }
    }
    else
    {
      rows = new int[ts->bcIdx.size()+1];
      cols = new int[ts->bcIdx.size()];
      a = new double[ts->bcIdx.size()];
      mm = scinew SparseRowMatrix(ts->bcIdx.size(),
				  meshH->nodesize(),
				  rows, cols,
				  ts->bcIdx.size(), a);
      for (i = 0; i<=ts->bcIdx.size()+1; i++)
      {
	rows[i] = i;
      }
    }
    omatH_ = mm;
    double *vals = mapping->get_rhs();
    int firstNode = 0;
    if (zeroTCL_.get())
    {
      cerr << "Skipping zero'th mesh node.\n";
      firstNode = 1;
    }
    if (m == "project")
    {
      if (p.size() > meshH->nodesize())
      {
	cerr << "Too many points to project (" << p.size() <<
	  " to " << meshH->nodesize() << ")\n";
	return;
      }

      Array1<int> selected(meshH->nodesize());
      selected.initialize(0);
      int counter = 0;
      for (int aa = 0; aa<p.size(); aa++)
      {
	double dt;
	int si = -1;
	double d;
	for (int bb = firstNode; bb<meshH->nodesize(); bb++)
	{
	  if (selected[bb]) continue;
	  dt = Vector(p[aa] - meshH->point(bb)).length2();
	  if (si==-1 || dt < d)
	  {
	    si = bb;
	    d = dt;
	  }
	}
	selected[si] = 1;

	if (!potMatTCL_.get() || aa!=0)
	{
	  a[counter] = 1;
	  cols[counter] = si;
	  counter++;
	}
	vals[aa] = si;
	ots->points.add(meshH->point(si));
      }
    }
  }
  else
  {
    cerr << "Unknown method: "<< m <<"\n";
    return;
  }
#endif

  omatrix_->send(omatrixH_);
  omat_->send(omatH_);
  osurf_->send(osurfH_);
}


} // End namespace SCIRun


