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

#include <PSECore/Datatypes/AttribPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<AttribHandle>::port_type("Attrib");
template<> clString SimpleIPort<AttribHandle>::port_color("Orange");

} // End namespace Datatypes
} // End namespace PSECore

