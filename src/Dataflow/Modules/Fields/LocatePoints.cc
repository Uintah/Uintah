
/*
 *  LocatePoints.cc:  Rescale a surface
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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {


class LocatePoints : public Module {
public:
  LocatePoints(const clString& id);
  virtual ~LocatePoints();
  virtual void execute();
private:
  FieldIPort*      imesh_;
  FieldIPort*    isurf_;
  MatrixOPort*     omat_;
  MatrixHandle     mat_handle_;
  int              mesh_gen_;
  int              surf_gen_;
};

extern "C" Module* make_LocatePoints(const clString& id)
{
  return new LocatePoints(id);
}

LocatePoints::LocatePoints(const clString& id)
  : Module("LocatePoints", id, Filter)
{
  imesh_=scinew FieldIPort(this, "Mesh", FieldIPort::Atomic);
  add_iport(imesh_);
  isurf_=scinew FieldIPort(this, "Surface", FieldIPort::Atomic);
  add_iport(isurf_);
  // Create the output port
  omat_=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
  add_oport(omat_);
  mesh_gen_=-1;
  surf_gen_=-1;
}

LocatePoints::~LocatePoints()
{
}

void sortpts(double *dist, int *idx) {
  double tmpd;
  int tmpi;
  int i,j;
  for (i=0; i<4; i++) {
    for (j=i+1; j<4; j++) {
      if (idx[j]<idx[i]) {
	tmpd=dist[i]; dist[i]=dist[j]; dist[j]=tmpd;
	tmpi=idx[i]; idx[i]=idx[j]; idx[j]=tmpi;
      }
    }
  }
}

void LocatePoints::execute()
{
  FieldHandle meshH;
  if(!imesh_->get(meshH))
    return;

  FieldHandle surfH;
  if (!isurf_->get(surfH))
    return;

  if (mesh_gen_ == meshH->generation && surf_gen_ == surfH->generation && 
      mat_handle_.get_rep()) {
    omat_->send(mat_handle_);
    return;
  }

  mesh_gen_ = meshH->generation;
  surf_gen_ = surfH->generation;

  //TriSurf *ts = surfH.get_rep();  // FIXME
  TriSurf *ts = 0;
  if (!ts) {
    cerr << "Error - need a TriSurface.\n";
    return;
  }

  int *rows=new int[ts->point_count()+1];
  int *cols=new int[ts->point_count()*4];
  double *a=new double[ts->point_count()*4];
// FIX_ME
#if 0 
  SparseRowMatrix *mm=scinew SparseRowMatrix(ts->point_count(),
					     meshH->nodesize(),
					     rows, cols,
					     ts->point_count()*4, a);
  mat_handle_=mm;
  int i,j;
  int ix;
  int nodes[4];
  double dist[4];
  double sum;
  for (i=0; i<ts->point_count(); i++) {
    rows[i]=i*4;
    if (!meshH->locate(&ix, ts->point(i), 1.e-4, 1.e-4)) {
      int foundIt=0;
      for (j=0; j<meshH->nodesize(); j++) {
	if ((meshH->point(j) - ts->point(i)).length2() < 0.001) {
	  ix=meshH->node(j).elems[0];
	  foundIt=1;
	  break;
	}
      }
      if (!foundIt) {
	cerr << "Error - couldn't find point "<<ts->point(i)<<" in mesh.\n";
	delete rows;
	delete cols;
	delete a;
	return;
      }
    }
    Element *e=meshH->element(ix);
    for (sum=0,j=0; j<4; j++) {
      nodes[j]=e->n[j];
      dist[j]=(ts->point(i) - meshH->point(nodes[j])).length();
      sum+=dist[j];
    }
    sortpts(dist, nodes);
    for (j=0; j<4; j++) {
      cols[i*4+j]=nodes[j];
      a[i*4+j]=dist[j]/sum;
    }
    if (i==ts->point_count()-1 || i==0 || (!(i%(ts->point_count()/10)))) {
      cerr << "i="<<i<<"\n";
      for (j=0; j<4; j++) {
	cerr << "a["<<i*4+j<<"]="<<a[i*4+j]<<" ";
      }
      cerr << "\n";
    }
  }
  rows[i]=i*4;
#endif 
  omat_->send(mat_handle_);
}

} // End namespace SCIRun


