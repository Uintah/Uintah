

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

#ifndef Packages/Uintah_Exceptions_TypeMismatchException_h
#define Packages/Uintah_Exceptions_TypeMismatchException_h

#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {
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
} // End namespace Uintah
   };

#endif


