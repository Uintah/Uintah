
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

#include <Datatypes/SimplePort.h>
#include <Datatypes/VectorField.h>

typedef Mailbox<SimplePortComm<VectorFieldHandle>*> _cfront_bug_VectorField_;
typedef SimpleIPort<VectorFieldHandle> VectorFieldIPort;
typedef SimpleOPort<VectorFieldHandle> VectorFieldOPort;

#endif
