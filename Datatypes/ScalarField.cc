
/*
 *  Field3D.cc: The Field3D Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ScalarField.h>
#include <Classlib/String.h>
#include <iostream.h>

PersistentTypeID ScalarField::type_id("ScalarField", "Datatype", 0);

ScalarField::ScalarField(Representation rep)
: have_bounds(0), have_minmax(0), rep(rep)
{
}

ScalarField::~ScalarField()
{
}

ScalarFieldRG* ScalarField::getRG()
{
    if(rep==RegularGridBase)
	return (ScalarFieldRG*)this;
    else
	return 0;
}

ScalarFieldUG* ScalarField::getUG()
{
    if(rep==UnstructuredGrid)
	return (ScalarFieldUG*)this;
    else
	return 0;
}

ScalarFieldRGBase* ScalarField::getRGBase()
{
cerr << "rep="<<rep<<"\n";
    if(rep==RegularGridBase)
	return (ScalarFieldRGBase*)this;
    else
	return 0;
}

void ScalarField::get_minmax(double& min, double& max)
{
    if(!have_minmax){
	compute_minmax();
	have_minmax=1;
    }
    min=data_min;
    max=data_max;
}

double ScalarField::longest_dimension()
{
    if(!have_bounds){
	compute_bounds();
	have_bounds=1;
	diagonal=bmax-bmin;
    }
    return Max(diagonal.x(), diagonal.y(), diagonal.z());
}

void ScalarField::get_bounds(Point& min, Point& max)
{
    if(!have_bounds){
	compute_bounds();
	have_bounds=1;
	diagonal=bmax-bmin;
    }
    max=bmax;
    min=bmin;
}

#define SCALARFIELD_VERSION 1

void ScalarField::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("ScalarField", SCALARFIELD_VERSION);
    int* repp=(int*)&rep;
    stream.io(*repp);
    if(stream.reading()){
	have_bounds=0;
	have_minmax=0;
    }
    stream.end_class();
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>
template class LockingHandle<ScalarField>;

#endif
