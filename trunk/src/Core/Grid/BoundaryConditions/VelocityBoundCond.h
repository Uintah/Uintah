#ifndef UINTAH_GRID_VelocityBoundCond_H
#define UINTAH_GRID_VelocityBoundCond_H

#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <SCIRun/Core/Geometry/Vector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <map>

namespace Uintah {
using namespace SCIRun;

using std::map;
   
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
     map<string,string> getMixed() const;
     virtual VelocityBoundCond* clone();
     
   private:
      std::map<string,string> d_comp_var;
     
   };

} // End namespace Uintah

#endif




