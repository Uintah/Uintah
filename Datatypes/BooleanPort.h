
/*
 *  BooleanPort.h
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_BooleanPort_h
#define SCI_project_BooleanPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/Boolean.h>

typedef Mailbox<SimplePortComm<sciBooleanHandle>*> _cfront_bug_sciBoolean_;
typedef SimpleIPort<sciBooleanHandle> sciBooleanIPort;
typedef SimpleOPort<sciBooleanHandle> sciBooleanOPort;

#endif
