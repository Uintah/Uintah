/*
 *  DiffFields.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Malloc/Allocator.h>

#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>

class DiffFields : public Module {
    ScalarFieldIPort *ifielda;
    ScalarFieldIPort *ifieldb;
    ScalarFieldOPort *ofield;
public:
    DiffFields(const clString& id);
    DiffFields(const DiffFields&, int deep);
    virtual ~DiffFields();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_DiffFields(const clString& id)
{
    return scinew DiffFields(id);
}
}

DiffFields::DiffFields(const clString& id)
: Module("DiffFields", id, Source)
{
    // Create the input port
    ifielda = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(ifielda);
    ifieldb = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(ifieldb);
    ofield = scinew ScalarFieldOPort(this, "SFRG",ScalarFieldIPort::Atomic);
    add_oport(ofield);
}

DiffFields::DiffFields(const DiffFields& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("DiffFields::DiffFields");
}

DiffFields::~DiffFields()
{
}

Module* DiffFields::clone(int deep)
{
    return scinew DiffFields(*this, deep);
}

void DiffFields::execute()
{
    ScalarFieldHandle sfIHa;
    ifielda->get(sfIHa);
    if (!sfIHa.get_rep()) return;
    ScalarFieldRG *isfa = sfIHa->getRG();
    if (!isfa) return;

    ScalarFieldHandle sfIHb;
    ifieldb->get(sfIHb);
    if (!sfIHb.get_rep()) return;
    ScalarFieldRG *isfb = sfIHb->getRG();
    if (!isfb) return;

    ScalarFieldRG* osf=new ScalarFieldRG;
    
    // make new field, and compute bbox and contents here!

    ScalarFieldHandle osfH(osf);
    ofield->send(osfH);
}
