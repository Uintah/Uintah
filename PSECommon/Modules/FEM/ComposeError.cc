//static char *id="@(#) $Id$";

/*
 *  ComposeError.cc: Evaluate the error in a finite element solution
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
#include <PSECore/CommonDatatypes/IntervalPort.h>
#include <SCICore/CoreDatatypes/Interval.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using SCICore::Containers::to_string;

class ComposeError : public Module {
    ScalarFieldIPort* upbound_field;
    ScalarFieldIPort* lowbound_field;
    IntervalIPort* intervalport;
    ScalarFieldOPort* outfield;
public:
    ComposeError(const clString& id);
    virtual ~ComposeError();
    virtual void execute();
};

Module* make_ComposeError(const clString& id) {
  return new ComposeError(id);
}

ComposeError::ComposeError(const clString& id)
: Module("ComposeError", id, Filter)
{
    // Create the output port
    lowbound_field=new ScalarFieldIPort(this, "Lower bound",
					ScalarFieldIPort::Atomic);
    add_iport(lowbound_field);
    upbound_field=new ScalarFieldIPort(this, "Upper bound",
				       ScalarFieldIPort::Atomic);
    add_iport(upbound_field);
    intervalport=new IntervalIPort(this, "Error Interval",
			       IntervalIPort::Atomic);
    add_iport(intervalport);

    outfield=new ScalarFieldOPort(this, "Solution",
				 ScalarFieldIPort::Atomic);
    add_oport(outfield);
}

ComposeError::~ComposeError()
{
}

void ComposeError::execute()
{
    ScalarFieldHandle lowf;
    if(!lowbound_field->get(lowf))
      return;
    ScalarFieldHandle upf;
    if(!upbound_field->get(upf))
      return;
    IntervalHandle interval;
    if(!intervalport->get(interval))
      return;

    ScalarFieldUG* lowfug=lowf->getUG();
    if(!lowfug){
	error("ComposeError can't deal with this field");
	return;
    }
    ScalarFieldUG* upfug=upf->getUG();
    if(!upfug){
	error("ComposeError can't deal with this field");
	return;
    }
    double* low=&lowfug->data[0];
    double* up=&upfug->data[0];
    if(upfug->mesh.get_rep() != lowfug->mesh.get_rep()){
        error("Two different meshes...\n");
	return;
    }

    int nelems=upfug->mesh->elems.size();
    ScalarFieldUG* outf=scinew ScalarFieldUG(ScalarFieldUG::ElementValues);
    outf->mesh=upfug->mesh;
    outf->data.resize(nelems);
    double mid=(interval->low+interval->high)/2.;
    int nlow=0;
    int nhigh=0;
    int nmid=0;
    for(int i=0;i<nelems;i++){
        if(low[i] > up[i]){
	    error("element "+to_string(i)+" has bad bounds: "+to_string(low[i])+", "+to_string(up[i]));
	}
	if(up[i] > interval->high){
	    outf->data[i]=up[i];
	    nhigh++;
	} else if(low[i] < interval->low){
	    outf->data[i]=low[i];
	    nlow++;
	} else {
	    outf->data[i]=mid;
	    nmid++;
	}
    }
    cerr << "low: " << nlow << endl;
    cerr << "mid: " << nmid << endl;
    cerr << "high: " << nhigh << endl;
    outfield->send(outf);
}
  
} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  1999/08/19 23:17:41  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:36  sparker
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
// Revision 1.1  1999/07/27 16:57:39  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:48  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
