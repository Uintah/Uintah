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

   class PressureBoundCond : public BoundCond  {
   public:
      PressureBoundCond(double& p);
      PressureBoundCond(ProblemSpecP& ps);
      virtual ~PressureBoundCond();
      virtual std::string getType() const;

      double  getPressure() const;
         
   private:
      PressureBoundCond(const PressureBoundCond&);
      PressureBoundCond& operator=(const PressureBoundCond&);
      
      double  d_press;
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1.2.1  2000/10/19 05:18:03  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.1  2000/10/18 03:39:48  jas
// Implemented Pressure boundary conditions.
//

#endif




