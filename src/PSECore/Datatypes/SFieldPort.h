// SFieldPort.h
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//

#ifndef SCI_project_SFieldPort_h
#define SCI_project_SFieldPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/SField.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;
using SCICore::Datatypes::SField;

typedef SimpleIPort<SFieldHandle> SFieldIPort;
typedef SimpleOPort<SFieldHandle> SFieldOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
