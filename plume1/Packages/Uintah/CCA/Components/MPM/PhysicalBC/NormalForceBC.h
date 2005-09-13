#ifndef UINTAH_GRID_NormalForceBC_H
#define UINTAH_GRID_NormalForceBC_H

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

using namespace SCIRun;
   
/**************************************

CLASS
   NormalForceBC
   
  
GENERAL INFORMATION

   NormalForceBC.h

   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NormalForceBC

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class NormalForceBC : public MPMPhysicalBC  {
   public:
      NormalForceBC(ProblemSpecP& ps);
      ~NormalForceBC();
      virtual std::string getType() const;

      // Get the load curve
      inline LoadCurve<double>* getLoadCurve() const {return d_loadCurve;}

      // Get the applied pressure at time t
      inline double getLoad(double t) const {return d_loadCurve->getLoad(t);}

      // Get the force per particle at time t
      double forcePerParticle(double time) const;

   private:
      NormalForceBC(const NormalForceBC&);
      NormalForceBC& operator=(const NormalForceBC&);
      
      // Load curve information (Pressure and time)
      LoadCurve<double>* d_loadCurve;

   };
} // End namespace Uintah

#endif
