//static char *id="@(#) $Id$";

/*
 *  cPhase.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/cVectorPort.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class cPhase : public Module {
    cVectorIPort* inport;
    ColumnMatrixOPort* outport;
    TCLdouble phase;
public:
    cPhase(const clString& id);
    virtual ~cPhase();
    virtual void execute();
};

Module* make_cPhase(const clString& id) {
  return new cPhase(id);
}

cPhase::cPhase(const clString& id)
: Module("cPhase", id, Filter), phase("phase", id, this)
{
    inport=scinew cVectorIPort(this, "Complex Vector", cVectorIPort::Atomic);
    add_iport(inport);
    outport=scinew ColumnMatrixOPort(this, "Real Vector", ColumnMatrixIPort::Atomic);
    add_oport(outport);
}

cPhase::~cPhase()
{
}

void cPhase::execute()
{
    cVectorHandle in;
    if(!inport->get(in))
	return;
    ColumnMatrixHandle out(new ColumnMatrix(in->size()));
    double ph=phase.get();
    double c=cos(ph);
    double s=sin(ph);
    int n=in->size();
    cVector& inh=*in.get_rep();
    ColumnMatrix& outh=*out.get_rep();
    for(int i=0;i<n;i++){
	cVector::Complex& cr(inh(i));
	double r=c*cr.real()+s*cr.imag();
	outh[i]=r;
    }
    outport->send(out);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/10/07 02:06:53  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:52  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:49  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:46  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:32  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:46  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
