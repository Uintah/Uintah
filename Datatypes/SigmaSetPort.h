
/*
 *  SigmaSetPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_project_SigmaSetPort_h
#define SCI_project_SigmaSetPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/SigmaSet.h>

typedef Mailbox<SimplePortComm<SigmaSetHandle>*> _cfront_bug_SigmaSet_;
typedef SimpleIPort<SigmaSetHandle> SigmaSetIPort;
typedef SimpleOPort<SigmaSetHandle> SigmaSetOPort;

#endif
