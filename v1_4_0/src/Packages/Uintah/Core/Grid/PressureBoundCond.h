#ifndef UINTAH_GRID_PressureBoundCond_H
#define UINTAH_GRID_PressureBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

using namespace SCIRun;
   
/**************************************

CLASS
   PressureBoundCond
   
   
GENERAL INFORMATION

   PressureBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PressureBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class PressureBoundCond : public BoundCond<double>  {
   public:
     PressureBoundCond(ProblemSpecP& ps,std::string& kind);
     virtual ~PressureBoundCond();
     virtual double  getValue() const;
     
   private:
#if 0
     PressureBoundCond(const PressureBoundCond&);
     PressureBoundCond& operator=(const PressureBoundCond&);
#endif
     
     double  d_press;
     
   };
} // End namespace Uintah

#endif
