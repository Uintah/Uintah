//  VField.cc - Vector Field
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/VField.h>

namespace SCIRun {

PersistentTypeID VField::type_id("VField", "Datatype", 0);
  
VField::VField() :
  Field()
{
}

VField::~VField()
{
}

void
VField::io(Piostream&)
{
}

} // End namespace SCIRun
