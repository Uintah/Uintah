// FieldWrapperPort.cc
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//


#include <PSECore/Datatypes/FieldWrapperPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<FieldWrapperHandle>::port_type("FieldWrapper");
template<> clString SimpleIPort<FieldWrapperHandle>::port_color("blue");

} // End namespace Datatypes
} // End namespace PSECore

