#ifndef UINTAH_GRID_KinematicBoundCond_H
#define UINTAH_GRID_KinematicBoundCond_H

#include <Uintah/Grid/BoundCond.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

namespace Uintah {

  using SCICore::Geometry::Vector;
   
/**************************************

CLASS
   KinematicBoundCond
   
   
GENERAL INFORMATION

   KinematicBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   KinematicBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class KinematicBoundCond : public BoundCond  {
   public:
      KinematicBoundCond(Vector& v);
      KinematicBoundCond(ProblemSpecP& ps);
      virtual ~KinematicBoundCond();
      virtual std::string getType() const;

      Vector getVelocity() const;
         
   private:
      KinematicBoundCond(const KinematicBoundCond&);
      KinematicBoundCond& operator=(const KinematicBoundCond&);
      
      Vector d_vel;
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




