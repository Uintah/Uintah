// AttribPort.cc
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//

#include <PSECore/Datatypes/DomainPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<DomainHandle>::port_type("Domain");
template<> clString SimpleIPort<DomainHandle>::port_color("green");

} // End namespace Datatypes
} // End namespace PSECore

