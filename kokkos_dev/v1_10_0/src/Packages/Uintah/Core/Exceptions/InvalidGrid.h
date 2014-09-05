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

#ifndef UINTAH_EXCEPTIONS_INVALIDGRID_H
#define UINTAH_EXCEPTIONS_INVALIDGRID_H

#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {

   class InvalidGrid : public SCIRun::Exception {
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
   };

} // End namespace Uintah

#endif


