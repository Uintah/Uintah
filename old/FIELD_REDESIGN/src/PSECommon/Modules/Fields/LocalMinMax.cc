//static char *id="@(#) $Id: LocalMinMax.cc,v";

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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Math/MinMax.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using SCICore::Math::Max;

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
	    int nnodes=mesh->nodes.size();
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
	    int nelems=mesh->elems.size();
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
	ScalarFieldRG *osfrg=new ScalarFieldRG;
	osfrg->resize(nx, ny, nz);
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1.2.1  2000/10/31 02:22:41  dmw
// Merging PSECommon changes from HEAD to FIELD_REDESIGN branch
//
// Revision 1.1  2000/10/29 04:34:51  dmw
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
