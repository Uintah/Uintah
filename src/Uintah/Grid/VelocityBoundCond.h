#ifndef UINTAH_GRID_VelocityBoundCond_H
#define UINTAH_GRID_VelocityBoundCond_H

#include <Uintah/Grid/BoundCond.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

namespace Uintah {

  using SCICore::Geometry::Vector;
   
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
      VelocityBoundCond(const VelocityBoundCond&);
      VelocityBoundCond& operator=(const VelocityBoundCond&);
      
      Vector d_vel;
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




