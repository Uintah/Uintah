// DomainPort.h
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//

#ifndef SCI_project_DomainPort_h
#define SCI_project_DomainPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/Domain.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;
using SCICore::Datatypes::DomainHandle;

typedef SimpleIPort<DomainHandle> DomainIPort;
typedef SimpleOPort<DomainHandle> DomainOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
