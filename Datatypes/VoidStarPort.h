
/*
 *  VoidStarPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_VoidStarPort_h
#define SCI_project_VoidStarPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/VoidStar.h>

typedef Mailbox<SimplePortComm<VoidStarHandle>*> _cfront_bug_VoidStar_;
typedef SimpleIPort<VoidStarHandle> VoidStarIPort;
typedef SimpleOPort<VoidStarHandle> VoidStarOPort;

#endif
