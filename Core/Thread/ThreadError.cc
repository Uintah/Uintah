
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

#include <Core/Thread/ThreadError.h>
namespace SCIRun {


ThreadError::ThreadError(const std::string& message)
    : message_(message)
{
}

ThreadError::ThreadError(const ThreadError& copy)
    : message_(copy.message_)
{
}

ThreadError::~ThreadError()
{
}

const char*
ThreadError::message() const
{
    return message_.c_str();
}

const char*
ThreadError::type() const
{
    return "ThreadError";
}


} // End namespace SCIRun
