/*
 *  VecSplit.cc: Compute and visualize error between two vectors
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

class VecSplit : public Module {
    ColumnMatrixIPort* ivecP;
    ColumnMatrixOPort* ovec1P;
    ColumnMatrixOPort* ovec2P;
    ColumnMatrixOPort* ovec3P;
public:
    VecSplit(const clString& id);
    virtual ~VecSplit();
    virtual void execute();
};

Module* make_VecSplit(const clString& id)
{
    return scinew VecSplit(id);
}

VecSplit::VecSplit(const clString& id)
: Module("VecSplit", id, Filter)
{
    // Create the input port
    ivecP=scinew ColumnMatrixIPort(this, "Input Vector",ColumnMatrixIPort::Atomic);
    add_iport(ivecP);
    ovec1P=scinew ColumnMatrixOPort(this, "Output Vector1",ColumnMatrixIPort::Atomic);
    add_oport(ovec1P);
    ovec2P=scinew ColumnMatrixOPort(this, "Output Vector2",ColumnMatrixIPort::Atomic);
    add_oport(ovec2P);
    ovec3P=scinew ColumnMatrixOPort(this, "Output Vector3",ColumnMatrixIPort::Atomic);
    add_oport(ovec3P);
}

VecSplit::~VecSplit() {
}

void VecSplit::execute()
{
     ColumnMatrixHandle ivecH;
     ColumnMatrix* ivec;
     if (!ivecP->get(ivecH) || !(ivec=ivecH.get_rep())) return;

     ColumnMatrix* ovec1, *ovec2, *ovec3;
     ovec1=new ColumnMatrix(6);
     ovec2=new ColumnMatrix(6);
     ovec3=new ColumnMatrix(6);

     double *o1, *o2, *o3, *i0;
     o1=ovec1->get_rhs();
     o2=ovec2->get_rhs();
     o3=ovec3->get_rhs();
     i0=ivec->get_rhs();

     for (int i=0; i<3; i++) {
	 o1[i]=o2[i]=o3[i]=i0[i];
	 o1[i+3]=o2[i+3]=o3[i+3]=0;
     }
     o1[3]=1;
     o2[4]=1;
     o3[5]=1;

     ColumnMatrixHandle ov1H(ovec1);
     ColumnMatrixHandle ov2H(ovec2);
     ColumnMatrixHandle ov3H(ovec3);

     ovec1P->send(ov1H);
     ovec2P->send(ov2H);
     ovec3P->send(ov3H);
}
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.1  1999/09/02 04:49:25  dmw
// more of Dave's modules
//
//
