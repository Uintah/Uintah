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

#include <Packages/Uintah/Core/Datatypes/TensorField.h>

#include <Dataflow/Ports/SimplePort.h>

namespace Uintah {

using namespace SCIRun;

typedef SimpleIPort<TensorFieldHandle> TensorFieldIPort;
typedef SimpleOPort<TensorFieldHandle> TensorFieldOPort;

} // End namespace Uintah

#endif
