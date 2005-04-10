#ifndef UINTAH_GRID_PressureBoundCond_H
#define UINTAH_GRID_PressureBoundCond_H

#include <Uintah/Grid/BoundCond.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

namespace Uintah {

  using SCICore::Geometry::Vector;
   
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
     PressureBoundCond(const PressureBoundCond&);
     PressureBoundCond& operator=(const PressureBoundCond&);
     
     double  d_press;
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//
// Revision 1.1  2000/10/18 03:39:48  jas
// Implemented Pressure boundary conditions.
//

#endif




