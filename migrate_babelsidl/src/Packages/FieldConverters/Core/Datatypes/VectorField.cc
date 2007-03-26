/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

#include <FieldConverters/Core/Datatypes/VectorField.h>

namespace FieldConverters {

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

} // end namespace FieldConverters
