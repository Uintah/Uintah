
/* REFERENCED */
static char *id="$Id$";

/*
 *  Runnable.h: The base class for all threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include "Runnable.h"
#include "Thread.h"
#include <iostream.h>

/*
 * This class should be a base class for any class which is to be
 * attached to a thread.  It provides a <i>run</i> virtual method
 * which can be overridden to provide the thread body.  When this
 * method returns, or the thread calls <i>Thread::exit</i>, the
 * thread terminates.  A <b>Runnable</b> should be attached to
 * only one thread.
 *
 * <p> It is very important that the <b>Runnable</b> object (or any
 * object derived from it) is never explicitly deleted.  It will be
 * deleted by the <b>Thread</b> to which it is attached, when the
 * thread terminates.  The destructor will be executed in the context
 * of this same thread.
 */

Runnable::Runnable()
{
    d_myThread=0;
}

Runnable::~Runnable()
{
    if(d_myThread){
        d_myThread->error("Runnable is being destroyed while thread is still running\n");
    }
}

//
// $Log$
// Revision 1.3  1999/08/25 02:37:59  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

