
/* REFERENCED */
static char *id="$Id$";

/*
 *  ThreadError: Exception class for unusual errors in the thread library
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: August 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Thread/ThreadError.h>

SCICore::Thread::ThreadError::ThreadError(const std::string& message)
    : d_message(message)
{
}

SCICore::Thread::ThreadError::~ThreadError()
{
}

std::string SCICore::Thread::ThreadError::message() const
{
    return d_message;
}

//
// $Log$
// Revision 1.3  1999/08/25 19:00:51  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
//
