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
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

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


