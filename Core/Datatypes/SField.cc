 //  SField.cc - Scalar Field
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/SField.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID SField::type_id("SField", "Datatype", 0);
  
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

} // end Datatypes
} // end SCICore
