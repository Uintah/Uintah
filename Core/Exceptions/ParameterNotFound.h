

/*
 *  ParameterNotFound.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Packages/Uintah_Exceptions_ParameterNotFound_h
#define Packages/Uintah_Exceptions_ParameterNotFound_h

#include <Packages/Uintah/Exceptions/ProblemSetupException.h>
#include <string>

namespace Uintah {
   public:
      ParameterNotFound(const std::string&);
      ParameterNotFound(const ParameterNotFound&);
      virtual ~ParameterNotFound();
      virtual const char* type() const;
   protected:
   private:
      ParameterNotFound& operator=(const ParameterNotFound&);
} // End namespace Uintah
   };

#endif


