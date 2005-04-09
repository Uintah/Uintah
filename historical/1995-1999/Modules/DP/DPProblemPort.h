
/*
 *  DPProblemPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DPProblemPort_h
#define SCI_project_DPProblemPort_h 1

#include <Datatypes/SimplePort.h>
#include <Modules/DP/DPProblem.h>

typedef Mailbox<SimplePortComm<DPProblemHandle>*> _cfront_bug_DPProblem_;
typedef SimpleIPort<DPProblemHandle> DPProblemIPort;
typedef SimpleOPort<DPProblemHandle> DPProblemOPort;

#endif
