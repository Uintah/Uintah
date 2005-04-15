//  VField.cc - Vector Field
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/VField.h>

namespace SCICore{
namespace Datatypes{

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

}  // end Datatypes
}  // end SCICore
