//static char *id="@(#) $Id:

/*
 *  Current.cc:  Compute the currents - I = del Phi * Sigma
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
// #include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>

using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class Current : public Module {
    ScalarFieldIPort* infield;
    MeshIPort* imesh;
    VectorFieldOPort* outfield;
public:
    TCLint interpolate;
    Current(const clString& id);
    virtual ~Current();
    virtual void execute();
};

Module* make_Current(const clString& id) {
  return new Current(id);
}

Current::Current(const clString& id)
: Module("Current", id, Filter), interpolate("interpolate", id, this)
{
    infield=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(infield);
    imesh = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(imesh);
    // Create the output port
    outfield=new VectorFieldOPort(this, "Geometry", VectorFieldIPort::Atomic);
    add_oport(outfield);
}

Current::~Current()
{
}

Vector tensorMult(const Vector &v, const Array1<double>& t) {
    Vector i;
    i.x(v.x()*t[0]+v.y()*t[1]+v.z()*t[2]);
    i.y(v.x()*t[1]+v.y()*t[3]+v.z()*t[4]);
    i.z(v.x()*t[2]+v.y()*t[4]+v.z()*t[5]);
    return i;
}

void Current::execute()
{
    ScalarFieldHandle sf;
    VectorFieldHandle vf;
    if(!infield->get(sf))
	return;
    ScalarFieldUG* sfug=sf->getUG();
    ScalarFieldRGBase* sfb=sf->getRGBase();
    MeshHandle meshH;
    if (!imesh->get(meshH) || !meshH.get_rep()) return;
    if (sfug) {
	VectorFieldUG* vfug;
	if(interpolate.get()){
	    vfug=new VectorFieldUG(VectorFieldUG::NodalValues);
	    vfug->mesh=sfug->mesh;
	    vfug->data.resize(sfug->data.size());
	    Mesh* mesh=sfug->mesh.get_rep();
	    int nnodes=mesh->nodes.size();
	    Array1<Vector>& gradients=vfug->data;
	    int i;
	    for(i=0;i<nnodes;i++)
		gradients[i]=Vector(0,0,0);
	    int nelems=mesh->elems.size();
	    for(i=0;i<nelems;i++){
		if(i%1000 == 0)
		    update_progress(i, nelems);
		Element* e=mesh->elems[i];
		Point pt;
		Vector grad1, grad2, grad3, grad4;
		/*double vol=*/mesh->get_grad(e, pt, grad1, grad2, grad3, grad4);
		double v1=sfug->data[e->n[0]];
		double v2=sfug->data[e->n[1]];
		double v3=sfug->data[e->n[2]];
		double v4=sfug->data[e->n[3]];
		Vector gradient(grad1*v1+grad2*v2+grad3*v3+grad4*v4);
		for(int j=0;j<4;j++){
		    gradients[e->n[j]]+=gradient;
		}
	    }
	    for(i=0;i<nnodes;i++){
		int idx;
		Vector v;
		if (!meshH->locate(meshH->nodes[i]->p, idx)) {
		    cerr << "Error - couldn't locate node "<<i<<" in mesh!\n";
		    idx=0;
		}
		Array1<double> t;
		t = meshH->cond_tensors[meshH->elems[idx]->cond];
		if(i%1000 == 0)
		    update_progress(i, nnodes);
		NodeHandle& n=mesh->nodes[i];
		gradients[i]*=1./(n->elems.size());
		gradients[i]=tensorMult(gradients[i], t);
	    }
	} else {
	    vfug=new VectorFieldUG(VectorFieldUG::ElementValues);
	    vfug->mesh=sfug->mesh;
	    Mesh* mesh=sfug->mesh.get_rep();
	    int nelems=mesh->elems.size();
	    vfug->data.resize(nelems);
	    for(int i=0;i<nelems;i++){
		//	    if(i%10000 == 0)
		//		update_progress(i, nelems);
		Element* e=mesh->elems[i];
		Point pt;
		Vector grad1, grad2, grad3, grad4;
		/*double vol=*/mesh->get_grad(e, pt, grad1, grad2, grad3, grad4);
		double v1=sfug->data[e->n[0]];
		double v2=sfug->data[e->n[1]];
		double v3=sfug->data[e->n[2]];
		double v4=sfug->data[e->n[3]];
		Vector gradient(grad1*v1+grad2*v2+grad3*v3+grad4*v4);
		Array1<double> t;
		t = meshH->cond_tensors[meshH->elems[i]->cond];
		vfug->data[i]=tensorMult(gradient, t);	
	    }
	}
	vf=vfug;
    } else {
	int nx=sfb->nx;
	int ny=sfb->ny;
	int nz=sfb->nz;
	VectorFieldRG *vfrg=new VectorFieldRG();
	vfrg->resize(nx, ny, nz);
	Point min, max;
	sfb->get_bounds(min, max);
	vfrg->set_bounds(min, max);
	ScalarFieldRGdouble *sfrd=sfb->getRGDouble();
	ScalarFieldRGfloat *sfrf=sfb->getRGFloat();
	ScalarFieldRGint *sfri=sfb->getRGInt();
	ScalarFieldRGshort *sfrs=sfb->getRGShort();
	ScalarFieldRGchar *sfrc=sfb->getRGChar();
//	ScalarFieldRGuchar *sfru=sfb->getRGUchar();
	for(int k=0;k<nz;k++){
	    update_progress(k, nz);
	    for(int j=0;j<ny;j++){
		for(int i=0;i<nx;i++){
		    Point p(sfb->get_point(i,j,k));
		    int idx;
		    Vector v;
		    if (!meshH->locate(p, idx)) {
			cerr << "Error - couldn't locate point "<<p<<" in mesh!\n";
			idx=0;
		    }
		    Array1<double> t;
		    t = meshH->cond_tensors[meshH->elems[idx]->cond];
		    if (sfrd) v=sfrd->gradient(i,j,k);
		    else if (sfrf) v=sfrf->gradient(i,j,k);
		    else if (sfri) v=sfri->gradient(i,j,k);
		    else if (sfrs) v=sfrs->gradient(i,j,k);
		    else if (sfrc) v=sfrc->gradient(i,j,k);
//		    else if (sfru) v=sfru->gradient(i,j,k);
		    else {
			cerr << "Unknown SFRG type in Current: "<<sfb->getType()<<".\n";
			return;
		    }
		    vfrg->grid(i,j,k)=tensorMult(v,t);
		    if (i==nx/2 && j==ny/2 && k==nz/2)
			cerr << "v="<<v<<"  c="<<vfrg->grid(i,j,k) << "  i="<<i<<"  j="<<j<<"  k="<<k<<"  t=["<<t[0]<<","<<t[1]<<","<<t[2]<<","<<t[3]<<","<<t[4]<<","<<t[5]<<"]\n";
		}
	    }
	}
	vf=vfrg;
    }
    outfield->send(vf);
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
