/*
 *  SFRGtoSFUG: Regular grid to unstructured grid - break hexes into tets
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Tester/RigorousTest.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;

/*
 * The values from Peters and DeMunck's 1991 papers:
 */

#define AIR_CONDUCTIVITY 	0.0
#define SKIN_CONDUCTIVITY 	1.0
#define BONE_CONDUCTIVITY 	0.05
#define CSF_CONDUCTIVITY 	4.620
#define GREY_CONDUCTIVITY 	1.0
#define WHITE_CONDUCTIVITY 	0.43

class SFRGtoSFUG : public Module {
    ScalarFieldIPort* ifld;
    ScalarFieldOPort* ofld;
    TCLint scalarAsCondTCL;
    TCLint removeAirTCL;
public:
    SFRGtoSFUG(const clString& id);
    virtual ~SFRGtoSFUG();
    virtual void execute();
    void setConductivities(Mesh *m);
    void genPtsAndTets(ScalarFieldRGBase *sf, ScalarFieldUG* sfug, 
		       int removeAir, int scalarAsCond);
};

Module* make_SFRGtoSFUG(const clString& id)
{
    return new SFRGtoSFUG(id);
}

SFRGtoSFUG::SFRGtoSFUG(const clString& id)
: Module("SFRGtoSFUG", id, Filter),
  scalarAsCondTCL("scalarAsCondTCL", id, this), 
  removeAirTCL("removeAirTCL", id, this)
{
    ifld=new ScalarFieldIPort(this, "SFRGin", ScalarFieldIPort::Atomic);
    add_iport(ifld);
    ofld=new ScalarFieldOPort(this, "SFUGout", ScalarFieldIPort::Atomic);
    add_oport(ofld);
}

SFRGtoSFUG::~SFRGtoSFUG() {
}



// we assume that the min and max of the field are at the middle of the first
// and last node respectively.
// for a field with nx=ny=nz=3 and min=(0,0,0), max=(1,1,1), the "corners"
// of the cells are at x, y, z positions: -0.25, 0.25, 0.75, 1.25
// note: this is consistent with the SegFldToSurfTree and CStoSFRG modules

void SFRGtoSFUG::genPtsAndTets(ScalarFieldRGBase *sf, ScalarFieldUG *sfug,
			       int removeAir, int scalarAsCond) {
    Mesh *mesh=sfug->mesh.get_rep();
    int offset=0;
    int nx, ny, nz;
    nx=sf->nx;
    ny=sf->ny;
    nz=sf->nz;
    double dmin, dmax;
    sf->get_minmax(dmin, dmax);
    if (dmin==48 && dmax<54 && (sf->getRGChar()||sf->getRGUchar())) offset=48;
    
    Point min, max;
    sf->get_bounds(min,max);
    Vector d(max-min);
    d.x(d.x()/(2.*(nx-1.)));
    d.y(d.y()/(2.*(ny-1.)));
    d.z(d.z()/(2.*(nz-1.)));
    min-=d;
    max+=d;
    Array3<int> nodes(nx+1, ny+1, nz+1);
    int currIdx=0;
    int i, j, k;
    cerr << "Starting node allocation...\n";
    for (i=0; i<nx+1; i++) {
	for (j=0; j<ny+1; j++) {
	    for (k=0; k<nz+1; k++) {
		nodes(i,j,k)=currIdx++;
		mesh->nodes.add(NodeHandle(new Node(min+Vector(d.x()*i, 
							       d.y()*j, 
							       d.z()*k))));
	    }
	}
    }
    cerr << "Done allocating nodes.\n";

    Array1<Element *> e(5);
    Array1<int> c(8);
    for (i=0; i<nx; i++) {
	for (int j=0; j<ny; j++) {
	    for (int k=0; k<nz; k++) {
		int cond=(int)(sf->get_value(i,j,k)-offset);
		if (!cond && removeAir) continue;
		c[0]=nodes(i,j,k);
		c[1]=nodes(i+1,j,k);
		c[2]=nodes(i+1,j+1,k);
		c[3]=nodes(i,j+1,k);
		c[4]=nodes(i,j,k+1);
		c[5]=nodes(i+1,j,k+1);
		c[6]=nodes(i+1,j+1,k+1);
		c[7]=nodes(i,j+1,k+1);
		if ((i+j+k)%2) {
		    e[0]=new Element(mesh, c[0], c[1], c[2], c[5]);
		    e[1]=new Element(mesh, c[0], c[2], c[3], c[7]);
		    e[2]=new Element(mesh, c[0], c[2], c[5], c[7]);
		    e[3]=new Element(mesh, c[0], c[4], c[5], c[7]);
		    e[4]=new Element(mesh, c[2], c[5], c[6], c[7]);
		} else {
		    e[0]=new Element(mesh, c[1], c[0], c[3], c[4]);
		    e[1]=new Element(mesh, c[1], c[3], c[2], c[6]);
		    e[2]=new Element(mesh, c[1], c[3], c[4], c[6]);
		    e[3]=new Element(mesh, c[1], c[5], c[4], c[6]);
		    e[4]=new Element(mesh, c[3], c[4], c[7], c[6]);
		}
		if (scalarAsCond)
		   e[0]->cond=e[1]->cond=e[2]->cond=e[3]->cond=e[4]->cond=cond;
		mesh->elems.add(e[0]); 
		mesh->elems.add(e[1]); 
		mesh->elems.add(e[2]); 
		mesh->elems.add(e[3]); 
		mesh->elems.add(e[4]); 
		sfug->data.add(cond);
		sfug->data.add(cond);
		sfug->data.add(cond);
		sfug->data.add(cond);
		sfug->data.add(cond);
	    }
	}
    }
}

void SFRGtoSFUG::setConductivities(Mesh *m) {
    m->cond_tensors.resize(6);
    m->cond_tensors[0].resize(6);
    m->cond_tensors[0].initialize(0);
    m->cond_tensors[0][0]=m->cond_tensors[0][3]=m->cond_tensors[0][5]=AIR_CONDUCTIVITY;

    m->cond_tensors[1].resize(6);
    m->cond_tensors[1].initialize(0);
    m->cond_tensors[1][0]=m->cond_tensors[1][3]=m->cond_tensors[1][5]=SKIN_CONDUCTIVITY;

    m->cond_tensors[2].resize(6);
    m->cond_tensors[2].initialize(0);
    m->cond_tensors[2][0]=m->cond_tensors[2][3]=m->cond_tensors[2][5]=BONE_CONDUCTIVITY;

    m->cond_tensors[3].resize(6);
    m->cond_tensors[3].initialize(0);
    m->cond_tensors[3][0]=m->cond_tensors[3][3]=m->cond_tensors[3][5]=CSF_CONDUCTIVITY;

    m->cond_tensors[4].resize(6);
    m->cond_tensors[4].initialize(0);
    m->cond_tensors[4][0]=m->cond_tensors[4][3]=m->cond_tensors[4][5]=GREY_CONDUCTIVITY;
    
    m->cond_tensors[5].resize(6);
    m->cond_tensors[5].initialize(0);
    m->cond_tensors[5][0]=m->cond_tensors[5][3]=m->cond_tensors[5][5]=WHITE_CONDUCTIVITY;
}

void SFRGtoSFUG::execute()
{
    ScalarFieldHandle ifldH;
    update_state(NeedData);

    if (!ifld->get(ifldH))
	return;
    if (!ifldH.get_rep()) {
	cerr << "Error: empty scalar field\n";
	return;
    }
    ScalarFieldRGBase *sfb;
    if (!(sfb=ifldH->getRGBase())) {
	cerr << "Error: field must be a regular grid\n";
	return;
    }
    update_state(JustStarted);

    int scalarAsCond = scalarAsCondTCL.get();
    int removeAir = removeAirTCL.get();
    
    Mesh *m = new Mesh;
    setConductivities(m);
    MeshHandle mH(m);
    ScalarFieldUG *sfug=new ScalarFieldUG(mH, ScalarFieldUG::ElementValues);

    genPtsAndTets(sfb, sfug, removeAir, scalarAsCond);
    m->pack_all();
    cerr << "Mesh has been built and output -- "<<m->nodes.size()<<" nodes, "<<m->elems.size()<<" elements.\n";
    for (int i=0; i<m->elems.size(); i++) {
	m->elems[i]->mesh = m;
	m->elems[i]->orient();
	m->elems[i]->compute_basis();
    }
    m->compute_neighbors();
    ofld->send(ScalarFieldHandle(sfug));
}

} // End namespace Modules
} // End namespace DaveW


// $Log:
