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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/VectorFieldPort.h>
#include <Datatypes/VectorFieldUG.h>
#include <Geometry/Point.h>

class Gradient : public Module {
    ScalarFieldIPort* infield;
    VectorFieldOPort* outfield;
public:
    Gradient(const clString& id);
    Gradient(const Gradient&, int deep);
    virtual ~Gradient();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_Gradient(const clString& id)
{
    return new Gradient(id);
}
};

Gradient::Gradient(const clString& id)
: Module("Gradient", id, Filter)
{
    infield=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(infield);
    // Create the output port
    outfield=new VectorFieldOPort(this, "Geometry", VectorFieldIPort::Atomic);
    add_oport(outfield);
}

Gradient::Gradient(const Gradient& copy, int deep)
: Module(copy, deep)
{
}

Gradient::~Gradient()
{
}

Module* Gradient::clone(int deep)
{
    return new Gradient(*this, deep);
}

void Gradient::execute()
{
    ScalarFieldHandle iff;
    if(!infield->get(iff))
	return;
    ScalarFieldUG* sfield=iff->getUG();
    if(!sfield){
	error("Gradient can't deal with this field");
	return;
    }
    VectorFieldUG* vfield=new VectorFieldUG;
    vfield->mesh=sfield->mesh;
    vfield->data.resize(sfield->data.size());
    Mesh* mesh=sfield->mesh.get_rep();
    int nnodes=mesh->nodes.size();
    Array1<Vector>& gradients=vfield->data;
    for(int i=0;i<nnodes;i++)
	gradients[i]=Vector(0,0,0);
    int nelems=mesh->elems.size();
    for(i=0;i<nelems;i++){
	update_progress(i, nelems);
	Element* e=mesh->elems[i];
	Point pt;
	Vector grad1, grad2, grad3, grad4;
	/*double vol=*/mesh->get_grad(e, pt, grad1, grad2, grad3, grad4);
	double v1=sfield->data[e->n[0]];
	double v2=sfield->data[e->n[1]];
	double v3=sfield->data[e->n[2]];
	double v4=sfield->data[e->n[3]];
	Vector gradient(grad1*v1+grad2*v2+grad3*v3+grad4*v4);
	for(int j=0;j<4;j++){
	    gradients[e->n[j]]+=gradient;
	}
    }
    for(i=0;i<nnodes;i++){
	update_progress(i, nnodes);
	NodeHandle& n=mesh->nodes[i];
	gradients[i]*=1./(n->elems.size());
    }
    outfield->send(VectorFieldHandle(vfield));
}
