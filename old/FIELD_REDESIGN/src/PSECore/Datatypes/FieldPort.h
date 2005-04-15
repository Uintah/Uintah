// FieldPort.h
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//

#ifndef SCI_project_FieldPort_h
#define SCI_project_FieldPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/Field.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;
using SCICore::Datatypes::Field;

typedef SimpleIPort<FieldHandle> FieldIPort;
typedef SimpleOPort<FieldHandle> FieldOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
