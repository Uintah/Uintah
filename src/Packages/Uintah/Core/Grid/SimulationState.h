#ifndef UINTAH_HOMEBREW_SimulationState_H
#define UINTAH_HOMEBREW_SimulationState_H

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <vector>
#include <iostream>

namespace Uintah {

using namespace SCIRun;
using std::cerr;
using std::endl;

class VarLabel;
class Material; 
class ICEMaterial;
class MPMMaterial;
   
    /**************************************
      
      CLASS
        SimulationState
      
        Short Description...
      
      GENERAL INFORMATION
      
        SimulationState.h
      
        Steven G. Parker
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2000 SCI Group
      
      KEYWORDS
        SimulationState
      
      DESCRIPTION
        Long description...
      
      WARNING
      
      ****************************************/
    
   class SimulationState : public RefCounted {
   public:
      SimulationState(ProblemSpecP &ps);
      ~SimulationState();
      const VarLabel* get_delt_label() const {
	 return delt_label;
      }

      void registerMPMMaterial(MPMMaterial*);
      void registerICEMaterial(ICEMaterial*);
      int getNumVelFields() const;

      int getNumMatls() const {
	 return (int)matls.size();
      }
      int getNumMPMMatls() const {
	 return (int)mpm_matls.size();
      }
      int getNumICEMatls() const {
	 return (int)ice_matls.size();
      }

      Material* getMaterial(int idx) const {
	 return matls[idx];
      }
      MPMMaterial* getMPMMaterial(int idx) const {
	 return mpm_matls[idx];
      }
      ICEMaterial* getICEMaterial(int idx) const {
	 return ice_matls[idx];
      }

      Vector getGravity() const {
	return d_gravity;
      }

      double getRefPress() const {
	return d_ref_press;
      }

      double getElapsedTime() const {
	return d_elapsed_time;
      }

      void setElapsedTime(double t) {
	d_elapsed_time = t;
      }
      bool d_mpm_cfd;
   private:

      void registerMaterial(Material*);

      SimulationState(const SimulationState&);
      SimulationState& operator=(const SimulationState&);
      
      const VarLabel* delt_label;
      std::vector<Material*> matls;
      std::vector<MPMMaterial*> mpm_matls;
      std::vector<ICEMaterial*> ice_matls;
      Vector d_gravity;
      double d_ref_press;
      double d_elapsed_time;
   };

} // End namespace Uintah

#endif
