//static char *id="@(#) $Id$";

/*
 *  FEMError.cc: Evaluate the error in a finite element solution
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <PSECore/CommonDatatypes/SurfacePort.h>
#include <SCICore/CoreDatatypes/ScalarFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Multitask/Task.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <values.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;
using namespace SCICore::Geometry;
using namespace SCICore::Multitask;

class FEMError : public Module {
    ScalarFieldIPort* infield;
    ScalarFieldOPort* upbound_field;
    ScalarFieldOPort* lowbound_field;
    Vector element_gradient(Element* e, ScalarFieldUG* field);

    ScalarFieldUG* lowf;
    ScalarFieldUG* upf;
    int np;
    Mesh* mesh;
    ScalarFieldUG* sfield;
public:
    FEMError(const clString& id);
    virtual ~FEMError();
    virtual void execute();
    void parallel(int);
};

Module* make_FEMError(const clString& id) {
  return new FEMError(id);
}

static void do_parallel(void* obj, int proc)
{
    FEMError* module=(FEMError*)obj;
    module->parallel(proc);
}
    
FEMError::FEMError(const clString& id)
: Module("FEMError", id, Filter)
{
    infield=new ScalarFieldIPort(this, "Solution",
				 ScalarFieldIPort::Atomic);
    add_iport(infield);
    // Create the output port
    lowbound_field=new ScalarFieldOPort(this, "Lower bound",
					ScalarFieldIPort::Atomic);
    add_oport(lowbound_field);
    upbound_field=new ScalarFieldOPort(this, "Upper bound",
				       ScalarFieldIPort::Atomic);
    add_oport(upbound_field);
}

FEMError::~FEMError()
{
}

void FEMError::parallel(int proc)
{
    int nelems=mesh->elems.size();
    int start_elem=nelems*proc/np;
    int end_elem=nelems*(proc+1)/np;
    double umin=MAXDOUBLE;
    double lmin=MAXDOUBLE;
    double umax=-MAXDOUBLE;
    double lmax=-MAXDOUBLE;
    for(int i=start_elem;i<end_elem;i++){
        Element* e=mesh->elems[i];
	Vector pv(0,0,0);
	Vector dv(0,0,0);
	int nneighbors=0;
	Point ecenter;
	double rad2;
	double err;
	// This computes the circumsphere...
	e->get_sphere2(ecenter, rad2, err);

	double totalvolume=0;
	Vector egrad(element_gradient(e, sfield));
	for(int j=0;j<4;j++){
	    if(e->face(j) != -1){
	        nneighbors++;
	        Element* ne=mesh->elems[e->face(j)];

		Point ncenter(ne->centroid());
		Vector ngrad(element_gradient(mesh->elems[e->face(j)], sfield));

		Vector dgrad(ngrad-egrad);
		Vector dcenter(ncenter-ecenter);
		if(dcenter.x() < 0){
		  dcenter.x(-dcenter.x());
		  dgrad.x(-dgrad.x());
		}
		if(dcenter.y() < 0){
		  dcenter.y(-dcenter.y());
		  dgrad.y(-dgrad.y());
		}
		if(dcenter.z() < 0){
		  dcenter.z(-dcenter.z());
		  dgrad.z(-dgrad.z());
		}
		pv+=dgrad*ne->volume();
		dv+=dcenter;
		totalvolume+=ne->volume();
	    }
	}
	dv*=1./nneighbors;
	Vector pvx(pv/(totalvolume*dv.x()));
	Vector pvy(pv/(totalvolume*dv.y()));
	Vector pvz(pv/(totalvolume*dv.z()));
	Vector vupper(Abs(pvx)+Abs(pvy)+Abs(pvz));
	double uu=vupper.x()+vupper.y()+vupper.z();
	Point& p0=mesh->nodes[e->n[0]]->p;
	double rad1=(e->centroid()-p0).length2();
	double upper=4*e->volume()*uu*uu*rad1;
	upf->data[i]=upper;
	umin=Min(upper, umin);
	umax=Max(upper, umax);

	Vector ccx(mesh->cond_tensors[e->cond][0],
		   mesh->cond_tensors[e->cond][1],
		   mesh->cond_tensors[e->cond][2]);
	Vector ccy(mesh->cond_tensors[e->cond][1],
		   mesh->cond_tensors[e->cond][3],
		   mesh->cond_tensors[e->cond][4]);
	Vector ccz(mesh->cond_tensors[e->cond][2],
		   mesh->cond_tensors[e->cond][4],
		   mesh->cond_tensors[e->cond][5]);
	double ll=3*(Dot(ccx, pvx)+Dot(ccy, pvy)+Dot(ccz, pvz))
	  /(ccx.x()+ccy.y()+ccz.z());
	double lower=4*e->volume()*ll*ll*rad1;
	lowf->data[i]=lower;
	lmin=Min(lower, lmin);
	lmax=Max(lower, lmax);
	if(proc == 0 && i%500 == 0)
	  update_progress(i, end_elem);
    }
}

void FEMError::execute()
{
    ScalarFieldHandle iff;
    if(!infield->get(iff))
	return;
    sfield=iff->getUG();
    if(!sfield){
	error("FEMError can't deal with this field");
	return;
    }
    mesh=sfield->mesh.get_rep();
    //int nnodes=mesh->nodes.size();
    int nelems=mesh->elems.size();
    upf=scinew ScalarFieldUG(ScalarFieldUG::ElementValues);
    upf->mesh=mesh;
    upf->data.resize(nelems);
    lowf=scinew ScalarFieldUG(ScalarFieldUG::ElementValues);
    lowf->mesh=mesh;
    lowf->data.resize(nelems);
    np=Task::nprocessors();
    Task::multiprocess(np, do_parallel, this);
    
    upbound_field->send(upf);
    lowbound_field->send(lowf);
}

Vector FEMError::element_gradient(Element* e, ScalarFieldUG* sfield)
{
    Vector grad1, grad2, grad3, grad4;
    Point pt;
    sfield->mesh->get_grad(e, pt, grad1, grad2, grad3, grad4);
    double v1=sfield->data[e->n[0]];
    double v2=sfield->data[e->n[1]];
    double v3=sfield->data[e->n[2]];
    double v4=sfield->data[e->n[3]];
    return grad1*v1+grad2*v2+grad3*v3+grad4*v4;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  1999/08/19 23:17:42  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:37  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:25  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:40  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:48  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
