
// $Id$

/*
 *  UnknownVariable.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Uintah_Exceptions_UnknownVariable_h
#define Uintah_Exceptions_UnknownVariable_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace Uintah {
   class UnknownVariable : public SCICore::Exceptions::Exception {
   public:
      UnknownVariable(const std::string& msg);
      UnknownVariable(const UnknownVariable&);
      virtual ~UnknownVariable();
      virtual const char* message() const;
      virtual const char* type() const;
   protected:
   private:
      std::string d_msg;
      UnknownVariable& operator=(const UnknownVariable&);
   };
}

#endif

//
// $Log$
// Revision 1.2  2000/04/26 06:48:43  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/11 07:10:46  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
//

