
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

#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

namespace Uintah {

   class InvalidValue : public ProblemSetupException {
   public:
      InvalidValue(const std::string&);
      InvalidValue(const InvalidValue&);
      virtual ~InvalidValue();
      virtual const char* type() const;
   protected:
   private:
      InvalidValue& operator=(const InvalidValue&);
   };

} // End namespace Uintah

#endif
