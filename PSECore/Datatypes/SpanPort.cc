
/*
 *  SpanPort.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <PSECore/Datatypes/SpanPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<SpanForestHandle>::port_type("Span");
template<> clString SimpleIPort<SpanForestHandle>::port_color("SteelBlue4");

} // End namespace Datatypes
} // End namespace PSECore



