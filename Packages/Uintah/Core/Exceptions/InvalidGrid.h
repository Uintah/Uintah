

/*
 *  InvalidGrid.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Packages/Uintah_Exceptions_InvalidGrid_h
#define Packages/Uintah_Exceptions_InvalidGrid_h

#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {
   public:
      InvalidGrid(const std::string& msg);
      InvalidGrid(const InvalidGrid&);
      virtual ~InvalidGrid();
      virtual const char* message() const;
      virtual const char* type() const;
   protected:
   private:
      std::string d_msg;
      InvalidGrid& operator=(const InvalidGrid&);
} // End namespace Uintah
   };

#endif


