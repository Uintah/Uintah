
/*
 *  DPGridPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DPGridPort_h
#define SCI_project_DPGridPort_h 1

#include <Datatypes/SimplePort.h>
#include <Modules/DP/DPGrid.h>

typedef Mailbox<SimplePortComm<DPGridHandle>*> _cfront_bug_DPGrid_;
typedef SimpleIPort<DPGridHandle> DPGridIPort;
typedef SimpleOPort<DPGridHandle> DPGridOPort;

#endif
