/*
 *  GenSigmaSet.cc:  
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Mar. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/SigmaSet.h>
#include <Datatypes/SigmaSetPort.h>
#include <Malloc/Allocator.h>

class GenSigmaSet : public Module {
    SigmaSetOPort *osigs;
    SigmaSetHandle s;

public:
    GenSigmaSet(const clString& id);
    GenSigmaSet(const GenSigmaSet&, int deep);
    virtual ~GenSigmaSet();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_GenSigmaSet(const clString& id)
{
   return scinew GenSigmaSet(id);
}
}

//static clString module_name("GenSigmaSet");

GenSigmaSet::GenSigmaSet(const clString& id)
: Module("GenSigmaSet", id, Source)
{
   // Create the output port
   osigs = scinew SigmaSetOPort(this, "Geometry", SigmaSetIPort::Atomic);
   add_oport(osigs);
   s = scinew SigmaSet(7,6);
   
   // Set up names of segmented materials
   s->names[0]="Air";
   s->names[1]="SKin";
   s->names[2]="Bone";
   s->names[3]="CSF";
   s->names[4]="Grey Matter";
   s->names[5]="White Matter";
   s->names[6]="Tumor";

   // Set up conductivity tensors
   s->vals(0,0)=s->vals(0,1)=s->vals(0,2)=s->vals(0,3)=s->vals(0,4)=
       s->vals(0,5)=0;		// conductivity through air
   s->vals(1,0)=s->vals(1,1)=s->vals(1,2)=s->vals(1,3)=s->vals(1,4)=
       s->vals(1,5)=.7;	 	// conductivity through skin
   s->vals(2,0)=s->vals(2,1)=s->vals(2,2)=s->vals(2,3)=s->vals(2,4)=
       s->vals(2,5)=.05;	// conductivity through bone   
   s->vals(3,0)=s->vals(3,1)=s->vals(3,2)=s->vals(3,3)=s->vals(3,4)=
       s->vals(3,5)=.5;		// conductivity through csf
   s->vals(4,0)=s->vals(4,1)=s->vals(4,2)=s->vals(4,3)=s->vals(4,4)=
       s->vals(4,5)=.6;		// conductivity through grey matter
   s->vals(5,0)=s->vals(5,1)=s->vals(5,2)=s->vals(5,3)=s->vals(5,4)=
       s->vals(5,5)=1.4;	// conductivity through white matter
   s->vals(6,0)=s->vals(6,1)=s->vals(6,2)=s->vals(6,3)=s->vals(6,4)=
       s->vals(6,5)=.4;		// conductivity through tumor
}

GenSigmaSet::GenSigmaSet(const GenSigmaSet& copy, int deep)
: Module(copy, deep)
{
   NOT_FINISHED("GenSigmaSet::GenSigmaSet");
}

GenSigmaSet::~GenSigmaSet()
{
}

Module* GenSigmaSet::clone(int deep)
{
   return scinew GenSigmaSet(*this, deep);
}

void GenSigmaSet::execute()
{
    osigs->send(s);
}
