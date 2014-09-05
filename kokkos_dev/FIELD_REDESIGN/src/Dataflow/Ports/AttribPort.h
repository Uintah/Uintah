// AttribPort.h
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//


#ifndef SCI_project_AttribPort_h
#define SCI_project_AttribPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/Attrib.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;
using SCICore::Datatypes::Attrib;

typedef SimpleIPort<AttribHandle> AttribIPort;
typedef SimpleOPort<AttribHandle> AttribOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
