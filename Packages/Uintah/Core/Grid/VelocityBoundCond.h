#ifndef UINTAH_GRID_VelocityBoundCond_H
#define UINTAH_GRID_VelocityBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
using namespace SCIRun;
   
/**************************************

CLASS
   VelocityBoundCond
   
   
GENERAL INFORMATION

   VelocityBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   VelocityBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class VelocityBoundCond : public BoundCond<Vector>  {
   public:
     VelocityBoundCond(ProblemSpecP& ps, const std::string& kind);
     virtual ~VelocityBoundCond();
     virtual Vector getValue() const;
     
   private:
#if 0
      VelocityBoundCond(const VelocityBoundCond&);
      VelocityBoundCond& operator=(const VelocityBoundCond&);
#endif
      
      Vector d_vel;
     
   };

} // End namespace Uintah

#endif




