
/*
 *  DPLinEqPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DPLinEqPort_h
#define SCI_project_DPLinEqPort_h 1

#include <Datatypes/SimplePort.h>
#include <Modules/DP/DPLinEq.h>

typedef Mailbox<SimplePortComm<DPLinEqHandle>*> _cfront_bug_DPLinEq_;
typedef SimpleIPort<DPLinEqHandle> DPLinEqIPort;
typedef SimpleOPort<DPLinEqHandle> DPLinEqOPort;

#endif
