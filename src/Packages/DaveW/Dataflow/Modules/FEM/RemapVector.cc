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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace DaveW {
using namespace SCIRun;

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
} // End namespace DaveW
}


