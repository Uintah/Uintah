#ifndef UINTAH_HOMEBREW_SimulationState_H
#define UINTAH_HOMEBREW_SimulationState_H

#include <Uintah/Grid/RefCounted.h>
#include <vector>
#include <iostream>
using std::cerr;
using std::endl;

namespace Uintah {
   class Material;
   class VarLabel;
   
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
      SimulationState();
      const VarLabel* get_delt_label() const {
	 return delt_label;
      }
      
      void registerMaterial(Material*);
      int getNumMatls() const {
	 return (int)matls.size();
      }
      int getNumVelFields() const {
	 cerr << "This needs to be fixed " << endl;
	 return (int)matls.size();
      }
      Material* getMaterial(int idx) const {
	 return matls[idx];
      }
   private:
      SimulationState(const SimulationState&);
      SimulationState& operator=(const SimulationState&);
      
      const VarLabel* delt_label;
      std::vector<Material*> matls;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.4  2000/04/26 06:48:55  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/04/25 22:29:33  guilkey
// Added method to return number of velocity fields.  Needs to be completed.
//
// Revision 1.2  2000/04/24 21:04:38  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.1  2000/04/20 18:56:30  sparker
// Updates to MPM
//
//

#endif

