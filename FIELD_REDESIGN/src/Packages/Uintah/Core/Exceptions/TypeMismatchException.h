
// $Id$

/*
 *  TypeMismatchException.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Uintah_Exceptions_TypeMismatchException_h
#define Uintah_Exceptions_TypeMismatchException_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace Uintah {
   class TypeMismatchException : public SCICore::Exceptions::Exception {
   public:
      TypeMismatchException(const std::string& msg);
      TypeMismatchException(const TypeMismatchException&);
      virtual ~TypeMismatchException();
      virtual const char* message() const;
      virtual const char* type() const;
   protected:
   private:
      std::string d_msg;
      TypeMismatchException& operator=(const TypeMismatchException&);
   };
}

#endif

//
// $Log$
// Revision 1.6  2000/04/26 06:48:42  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/11 07:10:46  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
//

