
/*
 *  GradientMagnitude.cc:  Unfinished modules
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
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Math/Expon.h>

namespace SCIRun {


class GradientMagnitude : public Module {
    VectorFieldIPort* infield;
    ScalarFieldOPort* outfield;
public:
    void parallel(int proc);
    VectorFieldHandle vf;
    ScalarFieldHandle sf;
    int np;
    GradientMagnitude(const clString& id);
    virtual ~GradientMagnitude();
    virtual void execute();
};

extern "C" Module* make_GradientMagnitude(const clString& id) {
  return new GradientMagnitude(id);
}

GradientMagnitude::GradientMagnitude(const clString& id)
    : Module("GradientMagnitude", id, Filter)
{
    infield=new VectorFieldIPort(this, "Vector", VectorFieldIPort::Atomic);
    add_iport(infield);

    // Create the output port
    outfield=new ScalarFieldOPort(this, "GradientMagnitude", ScalarFieldIPort::Atomic);
    add_oport(outfield);
}

GradientMagnitude::~GradientMagnitude()
{
}

void GradientMagnitude::parallel(int proc)
{
    ScalarFieldUG *sfug=sf->getUG();
    VectorFieldUG *vfug=vf->getUG();
    ScalarFieldRG *sfrg=sf->getRG();
    VectorFieldRG *vfrg=vf->getRG();
    if (sfug) {
	int sz=proc*vfug->data.size()/np;
	int ez=(proc+1)*vfug->data.size()/np;

	// won't bother with update_progress.  if we want it later, should
	// probably do loop unrolling here...
	for (int i=sz; i<ez; i++) {
	    sfug->data[i]=vfug->data[i].length();
	}
    } else {
	int nx=vfrg->nx;
	int ny=vfrg->ny;
	int nz=vfrg->nz;
	int sz=proc*nz/np;
	int ez=(proc+1)*nz/np;
	for(int k=sz;k<ez;k++){
	    if(proc == 0)
		update_progress(k-sz, ez-sz);
	    for(int j=0;j<ny;j++){
		for(int i=0;i<nx;i++){
		    sfrg->grid(i,j,k)=vfrg->grid(i,j,k).length();
		}
	    }
	}
    }
}

void GradientMagnitude::execute()
{
    if(!infield->get(vf))
	return;
    if (!vf.get_rep()) return;
    VectorFieldRG* vfrg=vf->getRG();
    VectorFieldUG* vfug=vf->getUG();
    if (vfrg) {
	ScalarFieldRG* sfrg=new ScalarFieldRG();
	sfrg->resize(vfrg->nx, vfrg->ny, vfrg->nz);
	Point min, max;
	vfrg->get_bounds(min, max);
	sfrg->set_bounds(min, max);
	sf=sfrg;
    } else {
	ScalarFieldUG::Type typ=ScalarFieldUG::NodalValues;
	if (vfug->typ == VectorFieldUG::ElementValues)
	    typ=ScalarFieldUG::ElementValues;
	ScalarFieldUG* sfug=new ScalarFieldUG(vfug->mesh, typ);
	sfug->data.resize(vfug->data.size());
	sf=sfug;
    }
    np=Thread::numProcessors();
    Thread::parallel(Parallel<GradientMagnitude>(this, &GradientMagnitude::parallel),
		     np, true);
    outfield->send(sf);
}

} // End namespace SCIRun

