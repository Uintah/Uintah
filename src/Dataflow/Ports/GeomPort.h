// GeomPort.h
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//


#ifndef SCI_project_GeomPort_h
#define SCI_project_GeomPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/Geom.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;
using SCICore::Datatypes::Geom;

typedef SimpleIPort<GeomHandle> GeomIPort;
typedef SimpleOPort<GeomHandle> GeomOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
