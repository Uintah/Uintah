
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/IntervalPort.h>
#include <Datatypes/Interval.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class ComposeError : public Module {
    ScalarFieldIPort* upbound_field;
    ScalarFieldIPort* lowbound_field;
    IntervalIPort* intervalport;
    ScalarFieldOPort* outfield;
public:
    ComposeError(const clString& id);
    ComposeError(const ComposeError&, int deep);
    virtual ~ComposeError();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ComposeError(const clString& id)
{
    return new ComposeError(id);
}
};

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

ComposeError::ComposeError(const ComposeError& copy, int deep)
: Module(copy, deep)
{
}

ComposeError::~ComposeError()
{
}

Module* ComposeError::clone(int deep)
{
    return new ComposeError(*this, deep);
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
  


