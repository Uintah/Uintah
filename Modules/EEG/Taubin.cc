/*
 *  Taubin.cc:  
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Mar. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrix.h>
#include <Datatypes/SparseRowMatrix.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/SurfacePort.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class Taubin : public Module {
    Array1<Array1<int> > nbrs;
    double lastGamma;
    double lastPb;
    TCLdouble gamma;
    TCLdouble pb;
    TCLint N;
    SurfaceIPort* isurf;
    SurfaceOPort* osurf;
    int tcl_exec;
    int reset;
    int init;
    int gen;
    SparseRowMatrix* srm;
    SparseRowMatrix* srg;
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
public:
    Taubin(const clString& id);
    Taubin(const Taubin&, int deep);
    virtual ~Taubin();
    virtual Module* clone(int deep);
    void bldMatrices();
    void bldCols();
    void bldNbrs();
    void smooth();
    void Reset();
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
};

extern "C" {
Module* make_Taubin(const clString& id)
{
   return scinew Taubin(id);
}
}

//static clString module_name("Taubin");

Taubin::Taubin(const clString& id)
: Module("Taubin", id, Source), gamma("gamma", id, this), pb("pb", id, this),
  N("N", id, this), tcl_exec(0), reset(0), init(0), gen(-1)
{
   // Create the input port
   isurf=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
   add_iport(isurf);
   // Create the output port
   osurf=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
   add_oport(osurf);
}

Taubin::Taubin(const Taubin& copy, int deep)
: Module(copy, deep), gamma("gamma", id, this), pb("pb", id, this),
  N("N", id, this), tcl_exec(0), reset(0), init(0), gen(-1)
{
   NOT_FINISHED("Taubin::Taubin");
}

Taubin::~Taubin()
{
}

Module* Taubin::clone(int deep)
{
   return scinew Taubin(*this, deep);
}

void Taubin::bldNbrs() {
    st->bldNodeInfo();
    nbrs.resize(st->nodeNbrs.size());
    for (int i=0; i<st->nodeNbrs.size(); i++) 
	nbrs[i].resize(st->nodeNbrs[i].size()+1);
//    st->nodeNbrs;
//    st->printNbrInfo();
    int j,k;
    for (i=0; i<st->nodeNbrs.size(); i++) {
//	cerr << "i="<<i<<"  nbrsSize="<<st->nodeNbrs[i].size();
	for (j=0, k=0; j<st->nodeNbrs[i].size(); j++, k++) {
	    if ((st->nodeNbrs[i][j]>i) && (j==k)) {
		nbrs[i][k]=i;
//		cerr << " "<<i;
		k++;
	    }
	    nbrs[i][k]=st->nodeNbrs[i][j];
//	    cerr << " "<<st->nodeNbrs[i][j];
	}
	if (j==k) {
	    nbrs[i][k]=i;
//	    cerr << " "<<i;
	}
//	cerr << "\n";
    }
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

void Taubin::bldMatrices() {
    int i,j;
    Array1<int> in_rows(nbrs.size()+1);
    Array1<int> in_cols;
    for (i=0; i<nbrs.size(); i++) {
	in_rows[i]=in_cols.size();
	for (j=0; j<nbrs[i].size(); j++) in_cols.add(nbrs[i][j]);
    }
    in_rows[i]=in_cols.size();
    srm=scinew SparseRowMatrix(nbrs.size(), in_cols.size(), in_rows, in_cols);
    srg=scinew SparseRowMatrix(nbrs.size(), in_cols.size(), in_rows, in_cols);

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

void Taubin::Reset() {
    int i;
    for (i=0; i<st->points.size(); i++) {
	oldX[i]=origX[i];
	oldY[i]=origY[i];
	oldZ[i]=origZ[i];
	st->points[i].x(oldX[i]);
	st->points[i].y(oldY[i]);
	st->points[i].z(oldZ[i]);
    }
}

void Taubin::bldCols() {
    int i;
    origX.resize(st->points.size());
    origY.resize(st->points.size());
    origZ.resize(st->points.size());
    oldX.resize(st->points.size());
    oldY.resize(st->points.size());
    oldZ.resize(st->points.size());
    tmpX.resize(st->points.size());
    tmpY.resize(st->points.size());
    tmpZ.resize(st->points.size());
    for (i=0; i<st->points.size(); i++) {
	origX[i]=oldX[i]=st->points[i].x();
	origY[i]=oldY[i]=st->points[i].y();
	origZ[i]=oldZ[i]=st->points[i].z();
    }
}

void Taubin::smooth() {
    int flops, memrefs;
    // multiply iteratively
    int iters=N.get();
    for (int iter=0; iter<iters; iter++) {
	srm->mult(oldX, tmpX, flops, memrefs);
	srm->mult(oldY, tmpY, flops, memrefs);
	srm->mult(oldZ, tmpZ, flops, memrefs);
	srg->mult(tmpX, oldX, flops, memrefs);
	srg->mult(tmpY, oldY, flops, memrefs);
	srg->mult(tmpZ, oldZ, flops, memrefs);
    }

    // copy the resultant points back into the data
    for (int i=0; i<st->points.size(); i++) {
	st->points[i].x(oldX[i]);
	st->points[i].y(oldY[i]);
	st->points[i].z(oldZ[i]);
    }    
}

void Taubin::execute()
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
	if (lastPb != pb.get() || lastGamma != gamma.get()) {
	    lastPb=pb.get();
	    lastGamma=gamma.get();
	    bldMatrices();
	}
	smooth();
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

void Taubin::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("Taubin needs a minor command");
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
