/*
 *  cMatrixPort.h
 *
 *  Written by:
 *   Leonid Zhukov
 *   Department of Computer Science
 *   University of Utah
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_cMatrixPort_h
#define SCI_project_cMatrixPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/cMatrix.h>

typedef Mailbox<SimplePortComm<cMatrixHandle>*> _cfront_bug_cMatrix_;
typedef SimpleIPort<cMatrixHandle> cMatrixIPort;
typedef SimpleOPort<cMatrixHandle> cMatrixOPort;

#endif
