//static char *id="@(#) $Id$";

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

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <PSECore/CommonDatatypes/SurfacePort.h>
#include <SCICore/CoreDatatypes/ScalarFieldRG.h>
#include <SCICore/CoreDatatypes/ScalarFieldUG.h>
#include <PSECore/CommonDatatypes/VectorFieldPort.h>
#include <SCICore/CoreDatatypes/VectorFieldRG.h>
#include <SCICore/CoreDatatypes/VectorFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Multitask/Task.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Multitask;

class GradientMagnitude : public Module {
    VectorFieldIPort* infield;
    ScalarFieldOPort* outfield;
public:
    void parallel(int proc);
    VectorFieldHandle vf;
    ScalarFieldHandle sf;
    int np;
    GradientMagnitude(const clString& id);
    GradientMagnitude(const GradientMagnitude&, int deep);
    virtual ~GradientMagnitude();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_GradientMagnitude(const clString& id) {
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

GradientMagnitude::GradientMagnitude(const GradientMagnitude& copy, int deep)
    : Module(copy, deep)
{
}

GradientMagnitude::~GradientMagnitude()
{
}

Module* GradientMagnitude::clone(int deep)
{
    return new GradientMagnitude(*this, deep);
}

static void do_parallel(void* obj, int proc)
{
    GradientMagnitude* module=(GradientMagnitude*)obj;
    module->parallel(proc);
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
    np=Task::nprocessors();
    Task::multiprocess(np, do_parallel, this);
    outfield->send(sf);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:28  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:42  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:11  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//
