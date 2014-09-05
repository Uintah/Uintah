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

#ifndef UINTAH_EXCEPTIONS_PARAMETERNOTFOUND_H
#define UINTAH_EXCEPTIONS_PARAMETERNOTFOUND_H

#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <string>

namespace Uintah {

   class ParameterNotFound : public ProblemSetupException {
   public:
      ParameterNotFound(const std::string&);
      ParameterNotFound(const ParameterNotFound&);
      virtual ~ParameterNotFound();
      virtual const char* type() const;
   protected:
   private:
      ParameterNotFound& operator=(const ParameterNotFound&);
   };

} // End namespace Uintah

#endif


