
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

#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Dataflow/Ports/cVectorPort.h>

namespace SCIRun {


class cPhase : public Module {
    cVectorIPort* inport;
    ColumnMatrixOPort* outport;
    TCLdouble phase;
public:
    cPhase(const clString& id);
    virtual ~cPhase();
    virtual void execute();
};

extern "C" Module* make_cPhase(const clString& id) {
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

} // End namespace SCIRun

