
/*
 *  InvalidValue.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_INVALIDVALUE_H
#define UINTAH_EXCEPTIONS_INVALIDVALUE_H

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

   using namespace SCIRun;

   class InvalidValue : public Exception {
   public:
      InvalidValue(const std::string&, const char* file, int line);
      InvalidValue(const InvalidValue&);
      virtual ~InvalidValue();
      virtual const char* message() const;
      virtual const char* type() const;
   protected:
   private:
      std::string d_msg;
      InvalidValue& operator=(const InvalidValue&);
   };

} // End namespace Uintah

#endif
