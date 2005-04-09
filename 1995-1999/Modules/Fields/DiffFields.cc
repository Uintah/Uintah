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
    
    // make new field, and compute bbox and contents here!
    
    //get bounds for compare...
    Point min_a, min_b, max_a, max_b;
    isfa->get_bounds(min_a, max_a);
    isfb->get_bounds(min_b, max_b);
    if ((min_a != min_b) && (max_a != max_b))
      {
	printf("DiffFields: Boundry not equal.\n  A: %f, %f, %f -> %f, %f, %f B: %f, %f, %f - %f, %f, %f\n",
	       min_a.x(),min_a.y(),min_a.z(),max_a.x(),max_a.y(),max_a.z(),min_b.x(),min_b.y(),min_b.z(),
	       max_b.x(),max_b.y(),max_b.z());
	return;
      }

    //Compare 3d array sizes...
    int dim1_a, dim2_a, dim3_a, dim1_b, dim2_b, dim3_b;
    dim1_a = isfa->grid.dim1();
    dim2_a = isfa->grid.dim2();
    dim3_a = isfa->grid.dim3();

    dim1_b = isfb->grid.dim1();
    dim2_b = isfb->grid.dim2();
    dim3_b = isfb->grid.dim3();

    if ( (dim1_a != dim1_b) || (dim2_a != dim2_b) || (dim3_a != dim3_b) )
      {
	printf("DiffFields: Dimensions of the two fields are not the same.\n A: %d x %d x %d B: %d x %d x %d\n",
	       dim1_a, dim2_a, dim3_a, dim1_b, dim2_b, dim3_b);
	return;
      }

    //Ok make the new field...
    ScalarFieldRG* diff_field=new ScalarFieldRG;
    diff_field->resize(dim1_a, dim2_a, dim3_a);
    diff_field->set_bounds(min_a, max_a);
    diff_field->grid.initialize(0);

    //And due the difference boy...
    for (int z = 0; z < dim3_a; z++)
      for (int y = 0; y < dim2_a; y++)
	for (int x = 0; x < dim1_a; x++)
	  {
	    diff_field->grid(x,y,z) = fabs(isfa->get_value(x,y,z) - isfb->get_value(x,y,z));
	  }
    
    ScalarFieldHandle diff_fieldH(diff_field);
    ofield->send(diff_fieldH);
}

