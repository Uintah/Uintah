#ifndef UINTAH_MPM_MPMPHYSICALBC_H
#define UINTAH_MPM_MPMPHYSICALBC_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
/**************************************

CLASS
   MPMPhysicalBC
   
GENERAL INFORMATION

   MPMPhysicalBC

   Honglai Tan
   Department of Materials Science and Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPMPhysicalBC

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MPMPhysicalBC  {
   public:
      MPMPhysicalBC() {};
      virtual ~MPMPhysicalBC() {};
      virtual std::string getType() const = 0;
         
   private:
      MPMPhysicalBC(const MPMPhysicalBC&);
      MPMPhysicalBC& operator=(const MPMPhysicalBC&);
   };
} // End namespace Uintah
   
#endif
