#ifndef UINTAH_GRID_ForceBC_H
#define UINTAH_GRID_ForceBC_H

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

using namespace SCIRun;
   
/**************************************

CLASS
   ForceBC
   
  
GENERAL INFORMATION

   ForceBC.h

   Honglai Tan
   Department of Materials Science and Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ForceBC

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ForceBC : public MPMPhysicalBC  {
   public:
      ForceBC(ProblemSpecP& ps);
      virtual std::string getType() const;

      const Vector&  getForceDensity() const;
      const Point&   getLowerRange() const;
      const Point&   getUpperRange() const;
         
   private:
      ForceBC(const ForceBC&);
      ForceBC& operator=(const ForceBC&);
      
      Vector d_forceDensity;
      Point  d_lowerRange;
      Point  d_upperRange;
   };
} // End namespace Uintah

#endif
