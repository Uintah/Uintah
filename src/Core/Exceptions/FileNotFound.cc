/*
 *  FileNotFound.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Exceptions/FileNotFound.h>

namespace SCIRun {

FileNotFound::FileNotFound(const std::string& message)
    : message_(message)
{
}

FileNotFound::FileNotFound(const FileNotFound& copy)
    : message_(copy.message_)
{
}

FileNotFound::~FileNotFound()
{
}

const char* FileNotFound::message() const
{
    return message_.c_str();
}

const char* FileNotFound::type() const
{
    return "FileNotFound";
}

} // End namespace SCIRun
