
/*
 *  ErrnoException.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Exceptions/ErrnoException.h>
#include <sstream>

using SCICore::Exceptions::ErrnoException;
using namespace std;

ErrnoException::ErrnoException(const std::string& message, int err)
   : d_errno(err)
{
   ostringstream str;
   const char* s = strerror(err);
   if(!s)
      s="(none)";
   str << message << " (errno=" << err << ": " << s << ")";
   d_message = str.str();
}

ErrnoException::ErrnoException(const ErrnoException& copy)
   : d_message(copy.d_message), d_errno(copy.d_errno)
{
}

ErrnoException::~ErrnoException()
{
}

const char* ErrnoException::message() const
{
   return d_message.c_str();
}

const char* ErrnoException::type() const
{
   return "SCICore::Exceptions::ErrnoException";
}

int ErrnoException::getErrno() const
{
   return d_errno;
}

//
// $Log$
// Revision 1.1.2.2  2000/10/26 17:51:52  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.2  2000/09/25 17:58:57  sparker
// Do not call variables errno due to #defines on some systems (linux)
// Correctly implemented copy CTORs
//
// Revision 1.1  2000/05/15 19:25:57  sparker
// Exception class for system calls (ones that use the errno variable)
//
//
