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

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

class RemapVector : public Module {
    ColumnMatrixIPort* irhsP;
    ColumnMatrixIPort* imapP;
    ColumnMatrixOPort* orhsP;
public:
    RemapVector(const clString& id);
    virtual ~RemapVector();
    virtual void execute();
};

Module* make_RemapVector(const clString& id)
{
    return scinew RemapVector(id);
}

RemapVector::RemapVector(const clString& id)
: Module("RemapVector", id, Filter)
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

     ColumnMatrix* orhs=scinew ColumnMatrix(nr);
     double *vals=orhs->get_rhs();
     ColumnMatrixHandle orhsH(orhs);

     for (int i=0; i<nr; i++) {
	 vals[i]=(*irhs)[(int)((*map)[i])];
     }

     orhsP->send(orhsH);
}
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.1  1999/09/02 04:49:25  dmw
// more of Dave's modules
//
//
