//static char *id="@(#) $Id$";

/*
 *  Gradient.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/Geometry/Point.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>

using std::cerr;

namespace SCIRun {

class Gradient : public Module {
    ScalarFieldIPort* infield;
    VectorFieldOPort* outfield;
public:
    TCLint interpolate;
    Gradient(const clString& id);
    virtual ~Gradient();
    virtual void execute();
};

extern "C" Module* make_Gradient(const clString& id) {
  return new Gradient(id);
}

Gradient::Gradient(const clString& id)
: Module("Gradient", id, Filter), interpolate("interpolate", id, this)
{
    infield=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(infield);
    // Create the output port
    outfield=new VectorFieldOPort(this, "Geometry", VectorFieldIPort::Atomic);
    add_oport(outfield);
}

Gradient::~Gradient()
{
}

void Gradient::execute()
{
    ScalarFieldHandle sf;
    VectorFieldHandle vf;
    if(!infield->get(sf))
	return;
    ScalarFieldUG* sfug=sf->getUG();
    ScalarFieldRGBase* sfb=sf->getRGBase();
    if (sfug) {
	VectorFieldUG* vfug;
	if (sfug->typ == ScalarFieldUG::NodalValues) {
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
		    if(i%1000 == 0)
			update_progress(i, nnodes);
		    NodeHandle& n=mesh->nodes[i];
		    gradients[i]*=1./(n->elems.size());
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
		    vfug->data[i]=gradient;
		}
	    }
	} else {
	    cerr << "Gradient: I don't know how to take element-value gradients.\n";
	    return;
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
		    if (sfrd)
			vfrg->grid(i,j,k)=sfrd->gradient(i,j,k);
		    else if (sfrf)
			vfrg->grid(i,j,k)=sfrf->gradient(i,j,k);
		    else if (sfri)
			vfrg->grid(i,j,k)=sfri->gradient(i,j,k);
		    else if (sfrs)
			vfrg->grid(i,j,k)=sfrs->gradient(i,j,k);
		    else if (sfrc)
			vfrg->grid(i,j,k)=sfrc->gradient(i,j,k);
//		    else if (sfru)
//			vfrg->grid(i,j,k)=sfru->gradient(i,j,k);
		    else {
			cerr << "Unknown SFRG type in Gradient: "<<sfb->getType()<<".\n";
			return;
		    }
		}
	    }
	}
	vf=vfrg;
    }
    outfield->send(vf);
}

} // End namespace SCIRun

