/*
 *  FieldFilter.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>
#include <stdio.h>

class FieldFilter : public Module {
    ScalarFieldIPort* ifield;
    ScalarFieldOPort* ofield;
    ScalarFieldHandle fldHandle;
    ScalarFieldRG* osf;
public:
    FieldFilter(const clString& id);
    FieldFilter(const FieldFilter&, int deep);
    virtual ~FieldFilter();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_FieldFilter(const clString& id)
{
    return new FieldFilter(id);
}
};

FieldFilter::FieldFilter(const clString& id)
: Module("FieldFilter", id, Filter)
{
    ifield=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(ifield);
    // Create the output port
    ofield=new ScalarFieldOPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_oport(ofield);
    osf=new ScalarFieldRG;
}

FieldFilter::FieldFilter(const FieldFilter& copy, int deep)
: Module(copy, deep)
{
}

FieldFilter::~FieldFilter()
{
}

Module* FieldFilter::clone(int deep)
{
    return new FieldFilter(*this, deep);
}

void FieldFilter::execute()
{
    ScalarFieldHandle ifh;
    if(!ifield->get(ifh))
	return;
    ScalarFieldRG* isf=ifh->getRG();
    if(!isf){
	error("FieldFilter can't deal with unstructured grids!");
	return;
    }
    ofield->send(osf);
}
