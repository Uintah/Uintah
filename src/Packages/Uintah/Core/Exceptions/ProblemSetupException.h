
// $Id$

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

#ifndef Uintah_Exceptions_ProblemSetupException_h
#define Uintah_Exceptions_ProblemSetupException_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace Uintah {
   class ProblemSetupException : public SCICore::Exceptions::Exception {
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
}

#endif

//
// $Log$
// Revision 1.5  2000/04/26 06:48:41  sparker
// Streamlined namespaces
//
// Revision 1.4  2000/04/11 07:10:44  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
//

