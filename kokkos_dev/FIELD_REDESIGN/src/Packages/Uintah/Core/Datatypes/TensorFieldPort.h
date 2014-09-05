
/*
 *  Texture3D.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_TensorFieldPort_h
#define SCI_project_TensorFieldPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include "TensorField.h"

namespace PSECore {
namespace Datatypes {

using SCICore::Datatypes::TensorFieldHandle;
typedef SimpleIPort<TensorFieldHandle> TensorFieldIPort;
typedef SimpleOPort<TensorFieldHandle> TensorFieldOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
