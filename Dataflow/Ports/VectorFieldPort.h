
/*
 *  VectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldPort_h
#define SCI_project_VectorFieldPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/VectorField.h>

namespace SCIRun {


typedef SimpleIPort<VectorFieldHandle> VectorFieldIPort;
typedef SimpleOPort<VectorFieldHandle> VectorFieldOPort;

} // End namespace SCIRun


#endif
