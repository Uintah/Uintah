/*
 *  InternalError.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Exceptions/InternalError.h>
namespace SCIRun {


InternalError::InternalError(const std::string& message)
    : d_message(message)
{
}

InternalError::InternalError(const InternalError& copy)
    : d_message(copy.d_message)
{
}

InternalError::~InternalError()
{
}

const char* InternalError::message() const
{
    return d_message.c_str();
}

const char* InternalError::type() const
{
    return "InternalError";
}

} // End namespace SCIRun
