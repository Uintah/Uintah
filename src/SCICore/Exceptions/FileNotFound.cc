/*
 *  FileNotFound.h: Generic exception for internal errors
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Exceptions/FileNotFound.h>

using SCICore::Exceptions::FileNotFound;

FileNotFound::FileNotFound(const std::string& message)
    : d_message(message)
{
}

FileNotFound::FileNotFound(const FileNotFound& copy)
    : d_message(copy.d_message)
{
}

FileNotFound::~FileNotFound()
{
}

const char* FileNotFound::message() const
{
    return d_message.c_str();
}

const char* FileNotFound::type() const
{
    return "SCICore::Exceptions::FileNotFound";
}

//
// $Log$
// Revision 1.1.2.3  2000/10/26 17:51:53  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.2  2000/09/25 19:46:41  sparker
// Quiet warnings under g++
//
// Revision 1.1  2000/05/20 08:06:12  sparker
// New exception classes
//
//
