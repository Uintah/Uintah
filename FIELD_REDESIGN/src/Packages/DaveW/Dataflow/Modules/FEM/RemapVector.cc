/*
 *  RemapVector.cc:  Remap a solution vector
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class RemapVector : public Module {
    ColumnMatrixIPort* irhsP;
    ColumnMatrixIPort* imapP;
    ColumnMatrixOPort* orhsP;
    TCLint zeroGround;
public:
    RemapVector(const clString& id);
    virtual ~RemapVector();
    virtual void execute();
};

extern "C" Module* make_RemapVector(const clString& id)
{
    return scinew RemapVector(id);
}

RemapVector::RemapVector(const clString& id)
: Module("RemapVector", id, Filter), zeroGround("zeroGround", id, this)
{
    // Create the input port
    irhsP=scinew ColumnMatrixIPort(this, "RHS in",ColumnMatrixIPort::Atomic);
    add_iport(irhsP);
    imapP=scinew ColumnMatrixIPort(this, "Map", ColumnMatrixIPort::Atomic);
    add_iport(imapP);

    // Create the output ports
    orhsP=scinew ColumnMatrixOPort(this,"RHS out",ColumnMatrixIPort::Atomic);
    add_oport(orhsP);
}

RemapVector::~RemapVector()
{
}

void RemapVector::execute()
{
     ColumnMatrixHandle irhsH;
     ColumnMatrix* irhs;
     if (!irhsP->get(irhsH) || !(irhs=irhsH.get_rep())) return;

     ColumnMatrixHandle mapH;
     ColumnMatrix *map;
     if (!imapP->get(mapH) || !(map=mapH.get_rep())) return;
     
     int nr=map->nrows();

     ColumnMatrixHandle orhsH;
     ColumnMatrix* orhs;
     if (zeroGround.get()) {
	 orhs=scinew ColumnMatrix(nr-1);
	 double *vals=orhs->get_rhs();
	 double v=(*irhs)[(int)((*map)[0])];
	 for (int i=1; i<nr; i++) {
	     vals[i-1]=(*irhs)[(int)((*map)[i])]-v;
	 }
     } else {
	 orhs=scinew ColumnMatrix(nr);
	 double *vals=orhs->get_rhs();
	 for (int i=0; i<nr; i++) {
	     vals[i]=(*irhs)[(int)((*map)[i])];
	 }
     }
     orhsH=orhs;
     orhsP->send(orhsH);
}
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.3  2000/03/17 09:25:44  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  1999/09/22 18:43:26  dmw
// added new GUI
//
// Revision 1.1  1999/09/02 04:49:25  dmw
// more of Dave's modules
//
//
