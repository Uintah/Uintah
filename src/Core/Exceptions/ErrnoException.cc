
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

#include <Core/Exceptions/ErrnoException.h>
#include <sstream>

namespace SCIRun {

using namespace std;

ErrnoException::ErrnoException(const std::string& message, int err)
   : errno_(err)
{
   ostringstream str;
   const char* s = strerror(err);
   if(!s)
      s="(none)";
   str << message << " (errno=" << err << ": " << s << ")";
   message_ = str.str();
}

ErrnoException::ErrnoException(const ErrnoException& copy)
   : message_(copy.message_), errno_(copy.errno_)
{
}

ErrnoException::~ErrnoException()
{
}

const char* ErrnoException::message() const
{
   return message_.c_str();
}

const char* ErrnoException::type() const
{
   return "ErrnoException";
}

int ErrnoException::getErrno() const
{
   return errno_;
}

} // End namespace SCIRun
