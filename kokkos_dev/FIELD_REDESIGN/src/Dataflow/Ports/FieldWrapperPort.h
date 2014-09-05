// FieldWrapperPort.h
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Group
//


#ifndef SCI_project_FieldWrapperPort_h
#define SCI_project_FieldWrapperPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/FieldWrapper.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;
using SCICore::Datatypes::FieldWrapperHandle;

typedef SimpleIPort<FieldWrapperHandle> FieldWrapperIPort;
typedef SimpleOPort<FieldWrapperHandle> FieldWrapperOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
