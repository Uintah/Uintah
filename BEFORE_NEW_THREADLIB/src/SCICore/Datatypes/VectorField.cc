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

#include <SCICore/Datatypes/VectorField.h>
#include <SCICore/Containers/String.h>

namespace SCICore {
namespace Datatypes {

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

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:44  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:31  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:20  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//
