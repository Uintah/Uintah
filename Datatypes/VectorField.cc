
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

#include <VectorField.h>
#include <Classlib/String.h>

PersistentTypeID VectorField::type_id("VectorField", "Datatype", 0);

VectorField::VectorField(Representation rep)
: rep(rep), have_bounds(0)
{
}

VectorField::~VectorField()
{
}

VectorFieldRG* VectorField::getRG()
{
    if(rep==RegularGrid)
	return (VectorFieldRG*)this;
    else
	return 0;
}

VectorFieldUG* VectorField::getUG()
{
    if(rep==UnstructuredGrid)
	return (VectorFieldUG*)this;
    else
	return 0;
}

double VectorField::longest_dimension()
{
    if(!have_bounds){
	compute_bounds();
	have_bounds=1;
	diagonal=bmax-bmin;
    }
    return Max(diagonal.x(), diagonal.y(), diagonal.z());
}

void VectorField::get_bounds(Point& min, Point& max)
{
    if(!have_bounds){
	compute_bounds();
	have_bounds=1;
	diagonal=bmax-bmin;
    }
    min=bmin;
    max=bmax;
}

#define VECTORFIELD_VERSION 1

void VectorField::io(Piostream& stream)
{
    int version=stream.begin_class("VectorField", VECTORFIELD_VERSION);
    int* repp=(int*)&rep;
    stream.io(*repp);
    if(stream.reading()){
	have_bounds=0;
    }
    stream.end_class();
}
