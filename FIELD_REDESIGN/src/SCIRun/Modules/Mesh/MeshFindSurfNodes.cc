
/*
 *  MeshFindSurfNodes.cc:  Rescale a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/SparseRowMatrix.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;

class MeshFindSurfNodes : public Module {
    MeshIPort* imesh;
    SurfaceIPort* isurf;
    MatrixOPort* omat;
    MatrixHandle matH;
    int meshGen;
    int surfGen;
public:
    MeshFindSurfNodes(const clString& id);
    virtual ~MeshFindSurfNodes();
    virtual void execute();
};

extern "C" Module* make_MeshFindSurfNodes(const clString& id)
{
    return new MeshFindSurfNodes(id);
}

MeshFindSurfNodes::MeshFindSurfNodes(const clString& id)
: Module("MeshFindSurfNodes", id, Filter)
{
    imesh=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(imesh);
    isurf=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurf);
    // Create the output port
    omat=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(omat);
    meshGen=-1;
    surfGen=-1;
}

MeshFindSurfNodes::~MeshFindSurfNodes()
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

void MeshFindSurfNodes::execute()
{
    MeshHandle meshH;
    if(!imesh->get(meshH))
	return;

    SurfaceHandle surfH;
    if (!isurf->get(surfH))
	return;

    if (meshGen == meshH->generation && surfGen == surfH->generation && matH.get_rep()) {
	omat->send(matH);
	return;
    }

    meshGen = meshH->generation;
    surfGen = surfH->generation;

    TriSurface *ts=dynamic_cast<TriSurface*>(surfH.get_rep());
    if (!ts) {
      cerr << "Error - need a TriSurface.\n";
      return;
    }

    int *rows=new int[ts->points.size()+1];
    int *cols=new int[ts->points.size()*4];
    double *a=new double[ts->points.size()*4];
    SparseRowMatrix *mm=scinew SparseRowMatrix(ts->points.size(),
					       meshH->nodes.size(),
					       rows, cols,
					       ts->points.size()*4, a);
    matH=mm;
    int i,j;
    int ix;
    int nodes[4];
    double dist[4];
    double sum;
    for (i=0; i<ts->points.size(); i++) {
      rows[i]=i*4;
      if (!meshH->locate(ts->points[i], ix, 1.e-4, 1.e-4)) {
	int foundIt=0;
	for (j=0; j<meshH->nodes.size(); j++) {
	  if ((meshH->nodes[j]->p-ts->points[i]).length2()<0.001) {
	    ix=meshH->nodes[j]->elems[0];
	    foundIt=1;
	    break;
	  }
	}
	if (!foundIt) {
	  cerr << "Error - couldn't find point "<<ts->points[i]<<" in mesh.\n";
	  delete rows;
	  delete cols;
	  delete a;
	  return;
	}
      }
      Element *e=meshH->elems[ix];
      for (sum=0,j=0; j<4; j++) {
	nodes[j]=e->n[j];
	dist[j]=(ts->points[i] - meshH->nodes[nodes[j]]->p).length();
	sum+=dist[j];
      }
      sortpts(dist, nodes);
      for (j=0; j<4; j++) {
	cols[i*4+j]=nodes[j];
	a[i*4+j]=dist[j]/sum;
      }
      if (i==ts->points.size()-1 || i==0 || (!(i%(ts->points.size()/10)))) {
	cerr << "i="<<i<<"\n";
	for (j=0; j<4; j++) {
	  cerr << "a["<<i*4+j<<"]="<<a[i*4+j]<<" ";
	}
	cerr << "\n";
      }
    }
    rows[i]=i*4;
    omat->send(matH);
}

} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.1.2.1  2000/10/31 02:33:17  dmw
// Merging SCIRun changes in HEAD into FIELD_REDESIGN branch
//
// Revision 1.1  2000/10/29 04:42:23  dmw
// MeshInterpVals -- fixed a bug
// MeshNodeComponent -- build a columnmatrix of the x/y/z position of the nodes
// MeshFindSurfNodes -- the surface nodes in a mesh
//
//
