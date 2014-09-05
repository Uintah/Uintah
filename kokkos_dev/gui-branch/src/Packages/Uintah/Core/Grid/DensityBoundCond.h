#ifndef UINTAH_GRID_DensityBoundCond_H
#define UINTAH_GRID_DensityBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
   
/**************************************

CLASS
   DensityBoundCond
   
   
GENERAL INFORMATION

   DensityBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   DensityBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class DensityBoundCond : public BoundCond<double>  {
   public:
     DensityBoundCond(ProblemSpecP& ps,std::string& kind);
     virtual ~DensityBoundCond();
     virtual double getValue() const;
     
   private:
#if 0
     DensityBoundCond(const DensityBoundCond&);
     DensityBoundCond& operator=(const DensityBoundCond&);
#endif
     
     double d_rho;
   };
} // End namespace Uintah

#endif
