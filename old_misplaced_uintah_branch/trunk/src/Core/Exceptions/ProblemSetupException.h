/*
 *  ProblemSetupException.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_PROBLEMSETUPEXCEPTION_H
#define UINTAH_EXCEPTIONS_PROBLEMSETUPEXCEPTION_H

#include <SCIRun/Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/Exceptions/uintahshare.h>
namespace Uintah {

   using namespace SCIRun;

   class UINTAHSHARE ProblemSetupException : public Exception {
   public:
      ProblemSetupException(const std::string& msg, const char* file, int line);
      ProblemSetupException(const ProblemSetupException&);
      virtual ~ProblemSetupException();
      virtual const char* message() const;
      virtual const char* type() const;
   private:
      std::string d_msg;
      ProblemSetupException& operator=(const ProblemSetupException&);
   };
} // End namespace Uintah

#endif


