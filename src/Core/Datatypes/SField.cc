 //  SField.cc - Scalar Field
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/SField.h>

namespace SCIRun {

PersistentTypeID SField::type_id("SField", "Field", 0);
  
SField::SField() :
  Field()
{
}

SField::~SField()
{
}


void
SField::io(Piostream&)
{
}

} // End namespace SCIRun
