
/*
 *  DPVecPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DPVecPort_h
#define SCI_project_DPVecPort_h 1

#include <Datatypes/SimplePort.h>
#include <Modules/DP/DPVec.h>

typedef Mailbox<SimplePortComm<DPVecHandle>*> _cfront_bug_DPVec_;
typedef SimpleIPort<DPVecHandle> DPVecIPort;
typedef SimpleOPort<DPVecHandle> DPVecOPort;

#endif
