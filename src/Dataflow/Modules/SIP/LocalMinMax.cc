
/*
 *  LocalMinMax.cc:  Compute a classified SF with 0=min, 1=non-extermal, 2=max
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Core/Math/MinMax.h>
#include <iostream>
using std::cerr;

namespace SCIRun {


class LocalMinMax : public Module {
    ScalarFieldIPort* ifield;
    ScalarFieldOPort* ofield;
public:
    LocalMinMax(const clString& id);
    virtual ~LocalMinMax();
    virtual void execute();
};

extern "C" Module* make_LocalMinMax(const clString& id) {
  return new LocalMinMax(id);
}

LocalMinMax::LocalMinMax(const clString& id)
: Module("LocalMinMax", id, Filter)
{
    ifield=new ScalarFieldIPort(this, " ", ScalarFieldIPort::Atomic);
    add_iport(ifield);
    // Create the output port
    ofield=new ScalarFieldOPort(this, " ", ScalarFieldIPort::Atomic);
    add_oport(ofield);
}

LocalMinMax::~LocalMinMax()
{
}

void LocalMinMax::execute()
{
    ScalarFieldHandle isfH;
    ScalarFieldHandle osfH;
    if(!ifield->get(isfH))
	return;
    ScalarFieldUG* isfug=isfH->getUG();
    ScalarFieldRG* isfrg=isfH->getRG();
    int minCnt=0;
    int maxCnt=0;
    int totalCnt=0;
    if (isfug) {
	ScalarFieldUG* osfug;
	Array1<int> nbrs;

//	Array1<int> nbrsTmp;

	if (isfug->typ == ScalarFieldUG::NodalValues) {
	    // look at the neighbors of each node
	    osfug=new ScalarFieldUG(ScalarFieldUG::NodalValues);
	    osfug->mesh=isfug->mesh;
	    osfug->data.resize(isfug->data.size());
	    osfug->data.initialize(0);
	    Mesh* mesh=isfug->mesh.get_rep();
	    int nnodes=mesh->nodesize();
	    totalCnt=nnodes;
	    int i,j;
	    for (i=0; i<nnodes; i++) {
		if (i && ((i%10000) == 0)) cerr << "LocalMinMax: "<<i<<"/"<<nnodes<<"\n";
		int debug=(i%28 == 0);
		debug=0;
		double myval=isfug->data[i];
		int min=1;
		int max=1;
		nbrs.resize(0);

		mesh->get_node_nbrhd(i, nbrs);

//		nbrsTmp.resize(0);
//		mesh->get_node_nbrhd(i, nbrsTmp);
//		for (j=0; j<nbrsTmp.size(); j++)
//		    mesh->get_node_nbrhd(nbrsTmp[j], nbrs);

		if (debug) {
		    cerr << "Node="<<i<<" nbrs=( ";
		    for (j=0; j<nbrs.size(); j++)
			cerr << nbrs[j] <<" ";
		    cerr <<")\n   myval="<<myval<<" nbrvals=( ";
		}
		for (j=0; j<nbrs.size(); j++) {
		    if (debug) cerr << isfug->data[nbrs[j]]<<" ";
		    if (isfug->data[nbrs[j]] < myval) min=0;
		    else if (isfug->data[nbrs[j]] > myval) max=0;
		}
		if (min) { osfug->data[i]=-1; minCnt++; }
		else if (max) { osfug->data[i]=1; maxCnt++; }
		if (debug) cerr << ")  min="<<min<<" max="<<max<< "v="<<osfug->data[i]<<"\n";
	    }
	} else {
	    // look at the neighbors of each element
	    osfug=new ScalarFieldUG(ScalarFieldUG::ElementValues);
	    osfug->mesh=isfug->mesh;
	    osfug->data.resize(isfug->data.size());
	    osfug->data.initialize(0);
	    Mesh* mesh=isfug->mesh.get_rep();
	    int nelems=mesh->elemsize();
	    totalCnt=nelems;
	    int i,j;
	    for (i=0; i<nelems; i++) {
		if (i && ((i%10000) == 0)) cerr << "LocalMinMax: "<<i<<"/"<<nelems<<"\n";
		int debug=(i%28 == 0);
		debug=0;
		double myval=isfug->data[i];
		int min=1;
		int max=1;
		nbrs.resize(0);

		mesh->get_elem_nbrhd(i, nbrs);

//		nbrsTmp.resize(0);
//		mesh->get_elem_nbrhd(i, nbrsTmp);
//		for (j=0; j<nbrsTmp.size(); j++)
//		    mesh->get_elem_nbrhd(nbrsTmp[j], nbrs);

		if (debug) {
		    cerr << "Elem="<<i<<" nbrs=( ";
		    for (j=0; j<nbrs.size(); j++)
			cerr << nbrs[j] <<" ";
		    cerr <<")\n   myval="<<myval<<" nbrvals=( ";
		}
		for (j=0; j<nbrs.size(); j++) {
		    if (debug) cerr << isfug->data[nbrs[j]]<<" ";
		    if (isfug->data[nbrs[j]] < myval) min=0;
		    else if (isfug->data[nbrs[j]] > myval) max=0;
		}
		if (min) { osfug->data[i]=-1; minCnt++; }
		else if (max) { osfug->data[i]=1; maxCnt++; }
		if (debug) cerr << ")  min="<<min<<" max="<<max<< "v="<<osfug->data[i]<<"\n";
	    }
	}
	osfH=osfug;
    } else {
	int nx=isfrg->nx;
	int ny=isfrg->ny;
	int nz=isfrg->nz;
	ScalarFieldRG *osfrg = new ScalarFieldRG(nx, ny, nz);
	osfrg->grid.initialize(0);
	Point min, max;
	isfrg->get_bounds(min, max);
	osfrg->set_bounds(min, max);
	totalCnt=nx*ny*nz;
	int cnt=0;
	for(int k=0;k<nz;k++){
	    update_progress(k, nz);
	    for(int j=0;j<ny;j++)
		for(int i=0;i<nx;i++,cnt++) {
		if (cnt && ((cnt%10000) == 0)) cerr << "LocalMinMax: "<<cnt<<"/"<<totalCnt<<"\n";
		    int min=1;
		    int max=1;
		    int ii,jj,kk;
		    double myval=isfrg->grid(i,j,k);
		    for (ii=Max(0,i-1); ii<i+1 && ii<nx; ii++)
			for (jj=Max(0,j-1); jj<j+1 && jj<ny; jj++)
			    for (kk=Max(0,k-1); kk<k+1 && kk<nz; kk++)
				if (isfrg->grid(ii,jj,kk) < myval) min=0;
				else if (isfrg->grid(ii,jj,kk) > myval) max=0;
		    if (min) { osfrg->grid(i,j,k) = -1; minCnt++; }
		    else if (max) { osfrg->grid(i,j,k) = 1; maxCnt++; }
		}
	}
	osfH=osfrg;
    }
    cerr << "LocalMinMax: "<<minCnt<<" minima and "<<maxCnt<<" maxima out of "<<totalCnt<<"\n";
    ofield->send(osfH);
}

} // End namespace SCIRun

