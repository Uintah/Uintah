#ifndef Uintah_MPM_MPMPhysicalBC_H
#define Uintah_MPM_MPMPhysicalBC_H

#include <string>

namespace Uintah {
namespace MPM {
   
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
   
} // end namespace MPM
} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2000/08/07 00:43:11  tan
// Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
// in MPM.  Currently implemented force boundary conditions.
//
//
