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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Math/Expon.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using SCICore::Thread::Parallel;
using SCICore::Thread::Thread;

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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/08/29 00:46:39  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/25 03:47:47  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:45  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:40  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
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
