// Field.cc - This is the base class from which all other fields are derived.
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include  <SCICore/Datatypes/Field.h>

namespace SCICore{
namespace Datatypes{

Field::Field(){
  // Set default values;
  status = NEW;
  elem_type = NODAL;

}

FieldInterface* Field::query_interface(const string& istring){
  // Nothing mathced, return NULL;
  return NULL;
}



}
}
