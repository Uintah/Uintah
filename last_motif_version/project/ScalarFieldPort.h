
/*
 *  ScalarField.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarFieldPort_h
#define SCI_project_ScalarFieldPort_h 1

#include <SimplePort.h>
#include <ScalarField.h>

typedef Mailbox<SimplePortComm<ScalarFieldHandle>*> _cfront_bug_ScalarField_;
typedef SimpleIPort<ScalarFieldHandle> ScalarFieldIPort;
typedef SimpleOPort<ScalarFieldHandle> ScalarFieldOPort;

#endif
