
/*
 *  MeshInterpVals.cc:  Rescale a surface
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

class MeshInterpVals : public Module {
    MeshIPort* imesh1;
    SurfaceIPort* isurf2;
    ColumnMatrixOPort* omatrix;
    TCLstring method;
    TCLint zeroTCL;
    ColumnMatrixHandle mapH;
    MatrixOPort* omat;
    MatrixHandle matH;

    int meshGen;
    int surfGen;
public:
    MeshInterpVals(const clString& id);
    virtual ~MeshInterpVals();
    virtual void execute();
};

Module* make_MeshInterpVals(const clString& id)
{
    return new MeshInterpVals(id);
}

MeshInterpVals::MeshInterpVals(const clString& id)
: Module("MeshInterpVals", id, Filter), method("method", id, this),
  zeroTCL("zeroTCL", id, this)
{
    imesh1=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(imesh1);
    isurf2=scinew SurfaceIPort(this, "Surface2", SurfaceIPort::Atomic);
    add_iport(isurf2);
    // Create the output port
    omat=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(omat);
    omatrix=scinew ColumnMatrixOPort(this, "Map", ColumnMatrixIPort::Atomic);
    add_oport(omatrix);
    meshGen=-1;
    surfGen=-1;
}

MeshInterpVals::~MeshInterpVals()
{
}

void MeshInterpVals::execute()
{
    MeshHandle meshH;
    if(!imesh1->get(meshH))
	return;

    SurfaceHandle surfH;
    if (!isurf2->get(surfH))
	return;

    if (meshGen == meshH->generation && surfGen == surfH->generation &&
	mapH.get_rep()) {
	omatrix->send(mapH);
	omat->send(matH);
	return;
    }

    meshGen = meshH->generation;
    surfGen = surfH->generation;
    clString m(method.get());

//    cerr << "Everything's good in MeshInterpVals so far...\n";

    if (m == "project") {	
	TriSurface *ts=scinew TriSurface;
	
	// first, set up the data point locations and values in an array
	Array1<Point> p;

	// get the right trisurface and grab those vals
	if(!(ts=surfH->getTriSurface())) {
	    cerr << "Error - need a trisurface!\n";
	    return;
	}
	if (ts->bcIdx.size()==0) {
	    ts->bcIdx.resize(ts->points.size());
	    ts->bcVal.resize(ts->points.size());
	    for (int ii=0; ii<ts->points.size(); ii++) {
		ts->bcIdx[ii]=ii;
		ts->bcVal[ii]=0;
	    }
	}
	int i;
	for (i=0; i<ts->bcIdx.size(); i++) {
	    p.add(ts->points[ts->bcIdx[i]]);
	}
	ColumnMatrix *map=scinew ColumnMatrix(ts->bcIdx.size());
	mapH=map;
	int *rows=new int[ts->bcIdx.size()+1];
	int *cols=new int[ts->bcIdx.size()];
	double *a=new double[ts->bcIdx.size()];
	SparseRowMatrix *mm=scinew SparseRowMatrix(ts->bcIdx.size(),
						   meshH->nodes.size(),
						   rows, cols,
						   ts->bcIdx.size(), a);
	for (i=0; i<ts->bcIdx.size()+1; i++) { rows[i]=i; }
	matH=mm;
//	cerr << "MeshInterpVals 1)...\n";
	double *vals=map->get_rhs();
//	cerr << "MeshInterpVals 2)...\n";
	int firstNode=0;
	if (zeroTCL.get()) {
	    cerr << "Skipping zero'th mesh node.\n";
	    firstNode=1;
	}
	if (m == "project") {
	    if (p.size() > meshH->nodes.size()) {
		cerr << "Too many points to project ("<<p.size()<<" to "<<meshH->nodes.size()<<")\n";
		return;
	    }

//	    cerr << "HERE ARE ALL THE PTS:\n";
//	    for (int ii=0; ii<ts->points.size(); ii++)
//		cerr << "  "<<ts->points[ii]<<"\n";

	    Array1<int> selected(meshH->nodes.size());
	    selected.initialize(0);
	    for (int aa=0; aa<p.size(); aa++) {
		double dt;
		int si=-1;
		double d;
		for (int bb=firstNode; bb<meshH->nodes.size(); bb++) {
		    if (selected[bb]) continue;
		    dt=Vector(p[aa]-meshH->nodes[bb]->p).length2();
		    if (si==-1 || dt<d) {
			si=bb;
			d=dt;
		    }
		}
		selected[si]=1;
//		cerr << "("<<aa<<") closest to "<<p[aa]<<"="<<meshH->nodes[si]->p<<"\n";
		vals[aa]=si;
		a[aa]=1;
		cols[aa]=si;
	    }
	}
    } else {
	cerr << "Unknown method: "<<m<<"\n";
	return;
    }
    omatrix->send(mapH);
    omat->send(matH);
}

} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.2  1999/10/07 02:08:20  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/05 01:15:28  dmw
// added all of the old SCIRun mesh modules
//
