//  Domain.cc - Manages sets of Attributes and Geometries
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/Domain.h>


namespace SCICore{
namespace Datatypes{


PersistentTypeID Domain::type_id("Domain", "Datatype", 0);

Domain::Domain()
{
}

Domain::~Domain()
{}

void Domain::io(Piostream&)
{}


}
}
