
/*
 *  TensorFieldPort.h: The TensorFieldPort datatype
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_Packages_DaveW_Datatypes_TensorFieldPort_h
#define SCI_Packages_DaveW_Datatypes_TensorFieldPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/DaveW/Core/Datatypes/General/TensorFieldBase.h>

namespace DaveW {
using namespace SCIRun;

typedef SimpleIPort<TensorFieldHandle> TensorFieldIPort;
typedef SimpleOPort<TensorFieldHandle> TensorFieldOPort;
} // End namespace DaveW



#endif
