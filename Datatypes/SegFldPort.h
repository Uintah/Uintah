
/*
 *  SegFld.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_SegFldPort_h
#define SCI_project_SegFldPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/SegFld.h>

typedef Mailbox<SimplePortComm<SegFldHandle>*> _cfront_bug_SegFld_;
typedef SimpleIPort<SegFldHandle> SegFldIPort;
typedef SimpleOPort<SegFldHandle> SegFldOPort;

#endif
