

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

#ifndef Packages/Uintah_Exceptions_InvalidValue_h
#define Packages/Uintah_Exceptions_InvalidValue_h

#include <Packages/Uintah/Exceptions/ProblemSetupException.h>
#include <string>

namespace Uintah {
   public:
      InvalidValue(const std::string&);
      InvalidValue(const InvalidValue&);
      virtual ~InvalidValue();
      virtual const char* type() const;
   protected:
   private:
      InvalidValue& operator=(const InvalidValue&);
} // End namespace Uintah
   };

#endif


