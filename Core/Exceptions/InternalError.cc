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
    : message_(message)
{
}

InternalError::InternalError(const InternalError& copy)
    : message_(copy.message_)
{
}

InternalError::~InternalError()
{
}

const char* InternalError::message() const
{
    return message_.c_str();
}

const char* InternalError::type() const
{
    return "InternalError";
}

} // End namespace SCIRun
