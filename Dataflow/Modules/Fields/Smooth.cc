
/*
 *  Smooth.cc:  
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Mar. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/SurfTree.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MusilRNG.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace SCIRun {

class Smooth : public Module {
    Array1<Array1<int> > nbrs;
    double lastGamma;
    double lastPb;
    TCLdouble gamma;
    TCLdouble pb;
    TCLdouble constraintTCL;
    TCLint N;
    TCLint constrainedTCL;
    TCLint jitterTCL;
    SurfaceIPort* isurf;
    SurfaceOPort* osurf;
    int tcl_exec;
    int reset;
    int init;
    int gen;
    SparseRowMatrix* srm;
    SparseRowMatrix* srg;
    double dx, dy, dz;
    ColumnMatrix oldX;
    ColumnMatrix oldY;
    ColumnMatrix oldZ;
    ColumnMatrix tmpX;
    ColumnMatrix tmpY;
    ColumnMatrix tmpZ;
    ColumnMatrix origX;
    ColumnMatrix origY;
    ColumnMatrix origZ;
    SurfaceHandle sh;
    SurfTree *st;
    MusilRNG mr;
public:
    Smooth(const clString& id);
    virtual ~Smooth();
    void bldMatrices();
    void bldCols();
    void bldNbrs();
    void smooth(int,double);
    void Reset();
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
};

extern "C" Module* make_Smooth(const clString& id)
{
   return scinew Smooth(id);
}

//static clString module_name("Smooth");

Smooth::Smooth(const clString& id)
: Module("Smooth", id, Source), gamma("gamma", id, this), pb("pb", id, this),
  N("N", id, this), tcl_exec(0), reset(0), init(0), gen(-1),
  constrainedTCL("constrainedTCL", id, this),
  constraintTCL("constraintTCL", id, this), jitterTCL("jitterTCL", id, this)
{
   // Create the input port
   isurf=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
   add_iport(isurf);
   // Create the output port
   osurf=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
   add_oport(osurf);
}

Smooth::~Smooth()
{
}

void Smooth::bldNbrs() {
    st->buildNodeInfo();
    nbrs.resize(st->nodeI_.size());
    int i;
    for (i=0; i<st->nodeI_.size(); i++) 
	nbrs[i].resize(st->nodeI_[i].nbrs.size()+1);
//    st->nodeNbrs;
//    st->printNbrInfo();
    int j,k;
    dx=dy=dz=0;
    for (i=0; i<st->nodeI_.size(); i++) {
//	cerr << "i="<<i<<"  nbrsSize="<<st->nodeNbrs[i].size();
	for (j=0, k=0; j<st->nodeI_[i].nbrs.size(); j++, k++) {
	    int nbr=st->nodeI_[i].nbrs[j];
	    if ((nbr>i) && (j==k)) {
		nbrs[i][k]=i;
		Vector v(st->points_[i]-st->points_[nbr]);
		if (Abs(v.x())>dx) dx=v.x();
		if (Abs(v.y())>dy) dy=v.y();
		if (Abs(v.z())>dz) dz=v.z();
//		cerr << " "<<i;
		k++;
	    }
	    nbrs[i][k]=nbr;
//	    cerr << " "<<st->nodeNbrs[i][j];
	}
	if (j==k) {
	    nbrs[i][k]=i;
//	    cerr << " "<<i;
	}
//	cerr << "\n";
    }
    
    cerr << "Smooth: dx="<<dx<<" dy="<<dy<<" dz="<<dz<<"\n";


    // these are the squares of the lengths of the axes of the ellipsoid 
    //   that the nodes can move within

    // (x-x0)^2 / dx^2 + (y-y0)^2 / dy^2 + (z-z0)^2 / dz^2 < 1.0

// go in and remove neighbors for some non-manifold cases

//    for (i=0; i<nbrs.size(); i++) {
//	cerr << i << "  ( ";
//	for (j=0; j<nbrs[i].size(); j++) {
//	    cerr << nbrs[i][j]<<" ";
//	}
//	cerr << ")  ( ";
//	for (j=0; j<st->nodeNbrs[i].size(); j++) {
//	    cerr << st->nodeNbrs[i][j]<<" ";
//	}
//	cerr << ")\n";
//    }
}

void Smooth::bldMatrices() {
    int i,j;
    Array1<int> in_rows(nbrs.size()+1);
    Array1<int> in_cols;
    for (i=0; i<nbrs.size(); i++) {
	in_rows[i]=in_cols.size();
	for (j=0; j<nbrs[i].size(); j++) in_cols.add(nbrs[i][j]);
    }
    in_rows[i]=in_cols.size();
    srm=scinew SparseRowMatrix(nbrs.size(), nbrs.size(), in_rows, in_cols);
    srg=scinew SparseRowMatrix(nbrs.size(), nbrs.size(), in_rows, in_cols);

    for (i=0; i<nbrs.size(); i++)
	if (nbrs[i].size()) {
	    double m=1./((1./lastGamma)-(1./lastPb));
	    double gn=lastGamma/(nbrs[i].size()-1);
	    double mn=m/(nbrs[i].size()-1);
	    for (j=0; j<nbrs[i].size(); j++) {
		srm->put(i,nbrs[i][j],gn);
		srg->put(i,nbrs[i][j],mn);
	    }
	    srm->put(i,i,1-lastGamma);
	    srg->put(i,i,1-m);
	}
}

void Smooth::Reset() {
    int i;
    for (i=0; i<st->points_.size(); i++) {
	oldX[i]=origX[i];
	oldY[i]=origY[i];
	oldZ[i]=origZ[i];
	st->points_[i].x(oldX[i]);
	st->points_[i].y(oldY[i]);
	st->points_[i].z(oldZ[i]);
    }
}

void Smooth::bldCols() {
    int i;
    origX.resize(st->points_.size());
    origY.resize(st->points_.size());
    origZ.resize(st->points_.size());
    oldX.resize(st->points_.size());
    oldY.resize(st->points_.size());
    oldZ.resize(st->points_.size());
    tmpX.resize(st->points_.size());
    tmpY.resize(st->points_.size());
    tmpZ.resize(st->points_.size());
    for (i=0; i<st->points_.size(); i++) {
	origX[i]=oldX[i]=st->points_[i].x();
	origY[i]=oldY[i]=st->points_[i].y();
	origZ[i]=oldZ[i]=st->points_[i].z();
    }
}
#if 0
void Smooth::smooth(int constrained, double cons) {
    double dx2=dx*.49*cons;
    dx2=dx2*dx2;
    double dy2=dy*.49*cons;
    dy2=dy2*dy2;
    double dz2=dz*.49*cons;
    dz2=dz2*dz2;
    int flops, memrefs;
    // multiplyiteratively
    int iters=N.get();

    if (jitterTCL.get()) {
	for (int i=0; i<st->points_.size(); i++) {
	    oldX[i] += (mr()-.5)*dx/10.;
	    oldY[i] += (mr()-.5)*dy/10.;
	    oldZ[i] += (mr()-.5)*dz/10.;
	}
    }
    for (int iter=0; iter<iters; iter++) {
	srm->mult(oldX, tmpX, flops, memrefs);
	srm->mult(oldY, tmpY, flops, memrefs);
	srm->mult(oldZ, tmpZ, flops, memrefs);
	srg->mult(tmpX, oldX, flops, memrefs);
	srg->mult(tmpY, oldY, flops, memrefs);
	srg->mult(tmpZ, oldZ, flops, memrefs);
    }

    // copy the resultant points back into the data
    for (int i=0; i<st->points_.size(); i++) {
	if (constrained) {
	    double tmp, d2;
	    // (x-x0)^2 / dx^2 + (y-y0)^2 / dy^2 + (z-z0)^2 / dz^2 < 1.0
	    tmp=oldX[i]-origX[i];
	    d2=tmp*tmp/dx2;
	    tmp=oldY[i]-origY[i];
	    d2+=tmp*tmp/dy2;
	    tmp=oldZ[i]-origZ[i];
	    d2+=tmp*tmp/dz2;
	    if (d2>1) {
		cerr << "Out-of-bounds (d2="<<d2<<") orig=("<<origX[i]<<", "<<origY[i]<<", "<<origZ[i]<<")\n\twants=("<<oldX[i]<<", "<<oldY[i]<<", "<<oldZ[i]<<")\n\tgot="; 
		// interesct the ellipsoid with the line between the 2 pts
		double t;
		double p1x, p1y, p1z, p0x, p0y, p0z;
		p0x=origX[i]; p1x=oldX[i];
		p0y=origY[i]; p1y=oldY[i];
		p0z=origZ[i]; p1z=oldZ[i];
		double denom=
		    dy2*dz2*(p1x*p1x+p0x*p0x)-
		    2*dy2*dz2*p1x*p0x+
		    dx2*dz2*(p1y*p1y+p0y*p0y)-
		    2*dx2*dz2*p1y*p0y+
		    dx2*dy2*(p1z*p1z+p0z*p0z)-
		    2*dx2*dy2*p1z*p0z;
		cerr << "[denom="<<denom<<"] ";
		t=Sqrt(denom*dx*dy*dz)/denom;
		oldX[i]=origX[i]+t*(oldX[i]-origX[i]);
		oldY[i]=origY[i]+t*(oldY[i]-origY[i]);
		oldZ[i]=origZ[i]+t*(oldZ[i]-origZ[i]);
		cerr << "("<<oldX[i]<<", "<<oldY[i]<<", "<<oldZ[i]<<")\n";
	    }
	}
	st->points_[i].x(oldX[i]);
	st->points_[i].y(oldY[i]);
	st->points_[i].z(oldZ[i]);
    }    
}
#endif

void Smooth::smooth(int constrained, double /* cons */) {
    int flops, memrefs;
    // multiplyiteratively
    int iters=N.get();

    if (jitterTCL.get()) {
	for (int i=0; i<st->points_.size(); i++) {
	    oldX[i] += (mr()-.5)*dx/10.;
	    oldY[i] += (mr()-.5)*dy/10.;
	    oldZ[i] += (mr()-.5)*dz/10.;
	}
    }
    for (int iter=0; iter<iters; iter++) {
	srm->mult(oldX, tmpX, flops, memrefs);
	srm->mult(oldY, tmpY, flops, memrefs);
	srm->mult(oldZ, tmpZ, flops, memrefs);
	srg->mult(tmpX, oldX, flops, memrefs);
	srg->mult(tmpY, oldY, flops, memrefs);
	srg->mult(tmpZ, oldZ, flops, memrefs);
    }

    // copy the resultant points back into the data
    for (int i=0; i<st->points_.size(); i++) {
	if (constrained) {

	  if (oldX[i]-origX[i] > 0.49*dx) oldX[i]=origX[i]+0.49*dx;
	  else if (origX[i]-oldX[i] > 0.49*dx) oldX[i]=origX[i]-0.49*dx;

	  if (oldY[i]-origY[i] > 0.49*dy) oldY[i]=origY[i]+0.49*dy;
	  else if (origY[i]-oldY[i] > 0.49*dy) oldY[i]=origY[i]-0.49*dy;

	  if (oldZ[i]-origZ[i] > 0.49*dz) oldZ[i]=origZ[i]+0.49*dz;
	  else if (origZ[i]-oldZ[i] > 0.49*dz) oldZ[i]=origZ[i]-0.49*dz;
	}
	st->points_[i].x(oldX[i]);
	st->points_[i].y(oldY[i]);
	st->points_[i].z(oldZ[i]);
    }    
}

void Smooth::execute()
{
//    if (!tcl_exec) return;
    if (!isurf->get(sh)) return;
    if (!sh.get_rep()) return;
    st=sh->getSurfTree();
    if (!st) {
	TriSurface *ts=sh->getTriSurface();
	if (ts) {
	    st=ts->toSurfTree();
	} else {
	    cerr << "Surface has to be a surfTree or a triSurface!\n";
	    return;
	}
    }
    cerr << "insurf gen="<<sh->generation<<"\n";
    if (!init || (sh->generation!=gen)) {
	init=1;
	bldNbrs();
	bldCols();
	lastPb=pb.get();
	lastGamma=gamma.get();
	bldMatrices();
	gen=sh->generation;
	tcl_exec=1;
    }
    if (!tcl_exec && !reset) return;
    if (reset) {
	Reset();
	bldNbrs();
	bldMatrices();
	reset=0;
    } else {
	tcl_exec=0;
	int constrained=constrainedTCL.get();
	cerr << "constrained="<<constrained<<"\n";
	if (lastPb != pb.get() || lastGamma != gamma.get()) {
	    lastPb=pb.get();
	    lastGamma=gamma.get();
	    bldMatrices();
	}
	double cons=constraintTCL.get();
	cerr << "cons="<<cons<<"\n";
	smooth(constrained, cons);
    }
    SurfaceHandle sh2;
    Array1<int> map, imap;
    if (sh->getTriSurface()) {
	TriSurface *ts=new TriSurface;
	st->extractTriSurface(ts, map, imap, 0);
	sh2=ts;
    } else {
	sh2=st;
    }
    osurf->send(sh2);
}

void Smooth::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("Smooth needs a minor command");
	return;
    }
    if (args[1] == "tcl_exec") {
	tcl_exec=1;
	want_to_execute();
    } else if (args[1] == "reset") {
	reset=1;
	want_to_execute();
    } else if (args[1] == "print") {
	int i,j;
	cerr << "Neighbors:\n";
	for (i=0; i<nbrs.size(); i++) {
	    if (nbrs[i].size()) {
		cerr << "  Point "<<i<<": ";
		for (j=0; j<nbrs[i].size(); j++)
		    cerr << nbrs[i][j]<<" ";
		cerr << "\n";
	    }
	}
	if (srm) srm->print();
    } else {
	Module::tcl_command(args, userdata);
    }
}
} // End namespace SCIRun
