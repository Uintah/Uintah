//static char *id="@(#) $Id$";

/*
 *  VectorField.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <CoreDatatypes/VectorField.h>
#include <Containers/String.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Math::Max;

PersistentTypeID VectorField::type_id("VectorField", "Datatype", 0);

VectorField::VectorField(Representation rep)
: have_bounds(0), rep(rep)
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

VectorFieldOcean* VectorField::getOcean()
{
    if(rep==OceanFile)
	return (VectorFieldOcean*)this;
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
    /*int version=*/stream.begin_class("VectorField", VECTORFIELD_VERSION);
    int* repp=(int*)&rep;
    stream.io(*repp);
    if(stream.reading()){
	have_bounds=0;
    }
    stream.end_class();
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:31  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:20  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//
