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

#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {

   using namespace SCIRun;

   class ProblemSetupException : public Exception {
   public:
      ProblemSetupException(const std::string& msg);
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


