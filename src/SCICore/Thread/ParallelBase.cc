
/* REFERENCED */
static char *id="$Id$";

/*
 *  ParallelBase: Helper class to instantiate several threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <SCICore/Thread/ParallelBase.h>

SCICore::Thread::ParallelBase::ParallelBase()
{
}

SCICore::Thread::ParallelBase::~ParallelBase()
{
}

//
// $Log$
// Revision 1.4  1999/08/25 19:00:49  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:37:57  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
