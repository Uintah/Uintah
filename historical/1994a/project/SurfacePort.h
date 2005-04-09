
/*
 *  SurfacePort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_SurfacePort_h
#define SCI_project_SurfacePort_h 1

#include <SimplePort.h>
#include <Surface.h>

typedef Mailbox<SimplePortComm<SurfaceHandle>*> _cfront_bug_Surface_;
typedef SimpleIPort<SurfaceHandle> SurfaceIPort;
typedef SimpleOPort<SurfaceHandle> SurfaceOPort;

#endif
