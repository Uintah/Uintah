/*
 *  ??????????????
 *
 *  Written by:
 *   ??????
 *   Department of Computer Science
 *   University of Utah
 *   March 199?
 *
 *  Copyright (C) 199? SCI Group
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
