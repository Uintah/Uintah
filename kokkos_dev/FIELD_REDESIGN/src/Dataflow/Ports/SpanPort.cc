
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

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/BBox.h>
#include <PSECore/Datatypes/SpanPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<SpanUniverseHandle>::port_type("SpanUniverse");
template<> clString SimpleIPort<SpanUniverseHandle>::port_color("SteelBlue4");

} // End namespace Datatypes
} // End namespace PSECore



