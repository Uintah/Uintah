// GeomPort.cc
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//


#include <PSECore/Datatypes/GeomPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<GeomHandle>::port_type("Geom");
template<> clString SimpleIPort<GeomHandle>::port_color("blue");

} // End namespace Datatypes
} // End namespace PSECore

