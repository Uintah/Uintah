// SFieldPort.cc
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//


#include <PSECore/Datatypes/SFieldPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<SFieldHandle>::port_type("SField");
template<> clString SimpleIPort<SFieldHandle>::port_color("yellow");

} // End namespace Datatypes
} // End namespace PSECore

