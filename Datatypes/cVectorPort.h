//
// cVectorPort.h
//
//  Written by:
//   Leonid Zhukov
//   Department of Computer Science
//   University of Utah
//   August 1997
//
//  Copyright (C) 1997 SCI Group
//

#ifndef SCI_project_cVectorPort_h
#define SCI_project_cVectorPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/cVector.h>

typedef Mailbox<SimplePortComm<cVectorHandle>*> _cfront_bug_cVector_;
typedef SimpleIPort<cVectorHandle> cVectorIPort;
typedef SimpleOPort<cVectorHandle> cVectorOPort;

#endif
