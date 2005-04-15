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
  
PersistentTypeID Field::type_id("Field", "Datatype", 0);

Field::Field()
  : status(NEW),
    data_loc(NODE)
{
}

Field::~Field()
{
}

void Field::io(Piostream&){
}

}
}
