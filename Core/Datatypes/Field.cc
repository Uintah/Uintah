//  Field.cc - This is the base class from which all other fields are derived.
//
//  Written by:
//   Eric Kuehne, Alexei Samsonov
//   Department of Computer Science
//   University of Utah
//   April 2000, December 2000
//
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/Field.h>

namespace SCIRun{

using std::cout;
using std::endl;

// GROUP: Persistence support
//////////
//

static Persistent* maker()
{
  return new Field();
}

PersistentTypeID Field::type_id("Field", "Datatype", maker);

//////////
// Persistent IO
#define FIELD_VERSION 1

void
Field::io(Piostream& stream)
{
  stream.begin_class("Field", FIELD_VERSION);

  stream.end_class();
}


//////////
// Constructors/Destructor
Field::Field()
{
}

Field::Field(const Field &)
{
}

Field::~Field()
{
}

}

