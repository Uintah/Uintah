
// $Id$

/*
 *  ErrnoException.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef SCICore_Exceptions_ErrnoException_h
#define SCICore_Exceptions_ErrnoException_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace SCICore {
   namespace Exceptions {
      class ErrnoException : public Exception {
      public:
	 ErrnoException(const std::string&, int err);
	 ErrnoException(const ErrnoException&);
	 virtual ~ErrnoException();
	 virtual const char* message() const;
	 virtual const char* type() const;
	 
	 int getErrno() const;
      protected:
      private:
	 std::string d_message;
	 int d_errno;

	 ErrnoException& operator=(const ErrnoException&);
      };
   }
}

#endif

//
// $Log$
// Revision 1.1.2.3  2000/10/26 17:51:52  moulding
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
