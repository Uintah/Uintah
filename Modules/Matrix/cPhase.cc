
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/cVectorPort.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class cPhase : public Module {
    cVectorIPort* inport;
    ColumnMatrixOPort* outport;
    TCLdouble phase;
public:
    cPhase(const clString& id);
    cPhase(const cPhase&, int deep);
    virtual ~cPhase();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_cPhase(const clString& id)
{
    return scinew cPhase(id);
}
};

cPhase::cPhase(const clString& id)
: Module("cPhase", id, Filter), phase("phase", id, this)
{
    inport=scinew cVectorIPort(this, "Complex Vector", cVectorIPort::Atomic);
    add_iport(inport);
    outport=scinew ColumnMatrixOPort(this, "Real Vector", ColumnMatrixIPort::Atomic);
    add_oport(outport);
}

cPhase::cPhase(const cPhase& copy, int deep)
: Module(copy, deep), phase("phase", id, this)
{
    NOT_FINISHED("cPhase::cPhase");
}

cPhase::~cPhase()
{
}

Module* cPhase::clone(int deep)
{
    return scinew cPhase(*this, deep);
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
	Complex& cr(inh(i));
	double r=c*cr.Re()+s*cr.Im();
	outh[i]=r;
    }
    outport->send(out);
}
