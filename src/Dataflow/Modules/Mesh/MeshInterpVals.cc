
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
    TCLint potMatTCL;
    ColumnMatrixHandle mapH;
    MatrixOPort* omat;
    MatrixHandle matH;
    SurfaceOPort *osurf;
    SurfaceHandle osurfH;
    int meshGen;
    int surfGen;
public:
    MeshInterpVals(const clString& id);
    virtual ~MeshInterpVals();
    virtual void execute();
};

extern "C" Module* make_MeshInterpVals(const clString& id)
{
    return new MeshInterpVals(id);
}

MeshInterpVals::MeshInterpVals(const clString& id)
: Module("MeshInterpVals", id, Filter), method("method", id, this),
  zeroTCL("zeroTCL", id, this), potMatTCL("potMatTCL", id, this)
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
    osurf=scinew SurfaceOPort(this, "NearestNodes", SurfaceIPort::Atomic);
    add_oport(osurf);
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
	osurf->send(osurfH);
	return;
    }

    meshGen = meshH->generation;
    surfGen = surfH->generation;
    clString m(method.get());

//    cerr << "Everything's good in MeshInterpVals so far...\n";
    
    TriSurface *ots=new TriSurface;
    osurfH=ots;
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
	int *rows;
	int *cols;
	double *a;
	SparseRowMatrix *mm;
	if (potMatTCL.get()) {
	    rows=new int[ts->bcIdx.size()];
	    cols=new int[(ts->bcIdx.size()-1)];
	    a=new double[(ts->bcIdx.size()-1)];
	    mm=scinew SparseRowMatrix(ts->bcIdx.size()-1,
				      meshH->nodes.size(),
				      rows, cols,
				      ts->bcIdx.size()-1, a);
	    for (i=0; i<ts->bcIdx.size(); i++) { rows[i]=i; }
	} else {
	    rows=new int[ts->bcIdx.size()+1];
	    cols=new int[ts->bcIdx.size()];
	    a=new double[ts->bcIdx.size()];
	    mm=scinew SparseRowMatrix(ts->bcIdx.size(),
				      meshH->nodes.size(),
				      rows, cols,
				      ts->bcIdx.size(), a);
	    for (i=0; i<=ts->bcIdx.size()+1; i++) { rows[i]=i; }
	}
	matH=mm;
//	cerr << "MeshInterpVals 1)...\n";
	double *vals=map->get_rhs();
//	cerr << "MeshInterpVals 2)...\n";
	int firstNode=0;
	if (zeroTCL.get()) {
	    cerr << "Skipping zero'th mesh node.\n";
	    firstNode=1;
	}
	int firstIdx;
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
	    int counter=0;
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
		if (!potMatTCL.get() || aa!=0) {
		    a[counter]=1;
		    cols[counter]=si;
		    counter++;
		}
		vals[aa]=si;
		ots->points.add(meshH->nodes[si]->p);
	    }
	}
    } else {
	cerr << "Unknown method: "<<m<<"\n";
	return;
    }
    omatrix->send(mapH);
    omat->send(matH);
    osurf->send(osurfH);
}

} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.6  2000/10/29 04:42:23  dmw
// MeshInterpVals -- fixed a bug
// MeshNodeComponent -- build a columnmatrix of the x/y/z position of the nodes
// MeshFindSurfNodes -- the surface nodes in a mesh
//
// Revision 1.5  2000/03/17 09:29:13  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  1999/12/11 05:48:55  dmw
// fixed code for generating the R matrix for RAinverse basis
//
// Revision 1.3  1999/12/10 06:58:13  dmw
// added another flag to MeshInterpVals
//
// Revision 1.2  1999/10/07 02:08:20  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/05 01:15:28  dmw
// added all of the old SCIRun mesh modules
//
