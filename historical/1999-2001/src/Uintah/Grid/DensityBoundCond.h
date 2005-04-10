#ifndef UINTAH_GRID_DensityBoundCond_H
#define UINTAH_GRID_DensityBoundCond_H

#include <Uintah/Grid/BoundCond.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

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
     DensityBoundCond(const DensityBoundCond&);
     DensityBoundCond& operator=(const DensityBoundCond&);
     
     double d_rho;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//
// Revision 1.1  2000/10/26 23:27:20  jas
// Added Density Boundary Conditions needed for ICE.
//
//

#endif




