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

#include <Core/Exceptions/ProblemSetupException.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/Exceptions/uintahshare.h>
namespace Uintah {

   class UINTAHSHARE ParameterNotFound : public ProblemSetupException {
   public:
      ParameterNotFound(const std::string&, const char* file, int line);
      ParameterNotFound(const ParameterNotFound&);
      virtual ~ParameterNotFound();
      virtual const char* type() const;
   protected:
   private:
      ParameterNotFound& operator=(const ParameterNotFound&);
   };

} // End namespace Uintah

#endif


