// FieldPort.cc
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//


#include <PSECore/Datatypes/FieldPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<FieldHandle>::port_type("Field");
template<> clString SimpleIPort<FieldHandle>::port_color("yellow");

} // End namespace Datatypes
} // End namespace PSECore

