
/*
 *  ThreadError: Exception class for unusual errors in the thread library
 *  $Id$
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

using SCICore::Thread::ThreadError;

ThreadError::ThreadError(const std::string& message)
    : d_message(message)
{
}

ThreadError::ThreadError(const ThreadError& copy)
    : d_message(copy.d_message)
{
}

ThreadError::~ThreadError()
{
}

const char*
ThreadError::message() const
{
    return d_message.c_str();
}

const char*
ThreadError::type() const
{
    return "SCICore::Thread::ThreadError";
}

//
// $Log$
// Revision 1.6  2000/03/23 20:43:14  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.5  2000/03/23 10:21:26  sparker
// Use libexc to print out stack straces on the SGI
// Added "name" method to ThreadError to match exception base class
// Fixed a compiler warning in Thread_irix.cc
//
// Revision 1.4  1999/08/28 03:46:51  sparker
// Final updates before integration with PSE
//
// Revision 1.3  1999/08/25 19:00:51  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
//
