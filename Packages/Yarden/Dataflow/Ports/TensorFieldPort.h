
/*
 *  TensorFieldPort.h: The TensorFieldPort datatype
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef SCI_Datatypes_TensorFieldPort_h
#define SCI_Datatypes_TensorFieldPort_h 

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/Yarden/Core/Datatypes/TensorField.h>

namespace Yarden {
using namespace SCIRun;
    
    typedef SimpleIPort<TensorFieldHandle> TensorFieldIPort;
    typedef SimpleOPort<TensorFieldHandle> TensorFieldOPort;
} // End namespace Yarden
    


#endif
