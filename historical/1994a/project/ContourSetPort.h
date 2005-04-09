
/*
 *  ContourSetPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ContourSetPort_h
#define SCI_project_ContourSetPort_h 1

#include <SimplePort.h>
#include <ContourSet.h>

typedef Mailbox<SimplePortComm<ContourSetHandle>*> _cfront_bug_ContourSet_;
typedef SimpleIPort<ContourSetHandle> ContourSetIPort;
typedef SimpleOPort<ContourSetHandle> ContourSetOPort;

#endif
