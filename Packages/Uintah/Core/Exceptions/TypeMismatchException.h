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

#ifndef UINTAH_EXCEPTIONS_TYPEMISMATCHEXCEPTION_H
#define UINTAH_EXCEPTIONS_TYPEMISMATCHEXCEPTION_H

#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {

   class TypeMismatchException : public SCIRun::Exception {
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
} // End namespace Uintah

#endif


