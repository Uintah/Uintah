#ifndef UINTAH_HOMEBREW_SimulationState_H
#define UINTAH_HOMEBREW_SimulationState_H

#include <Uintah/Grid/RefCounted.h>
#include <vector>
#include <iostream>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Grid/Material.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>

using std::cerr;
using std::endl;
using SCICore::Math::Max;
using SCICore::Geometry::Vector;
using Uintah::Material;
using SCICore::Geometry::Vector;

namespace Uintah {

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
      SimulationState(ProblemSpecP &ps);
      ~SimulationState();
      const VarLabel* get_delt_label() const {
	 return delt_label;
      }
      
      void registerMaterial(Material*);
      int getNumMatls() const {
	 return (int)matls.size();
      }
      int getNumVelFields() const {
	int num_vf=0;
	for (int i = 0; i < (int)matls.size(); i++) {
	  num_vf = Max(num_vf,matls[i]->getVFIndex());
	}
	return num_vf+1;
      }
      Material* getMaterial(int idx) const {
	 return matls[idx];
      }


      Vector getGravity() const {
	return d_gravity;
      }

   private:
      SimulationState(const SimulationState&);
      SimulationState& operator=(const SimulationState&);
      
      const VarLabel* delt_label;
      std::vector<Material*> matls;
      Vector d_gravity;

   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.12  2000/09/28 23:22:01  jas
// Added (int) to remove g++ warnings for STL size().  Reordered initialization
// to coincide with *.h declarations.
//
// Revision 1.11  2000/08/09 03:18:05  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.10  2000/06/27 20:14:09  guilkey
// Removed Kinetic and Strain energy labels and associated stuff.
//
// Revision 1.9  2000/05/31 22:27:52  guilkey
// Added stuff for integrated quanities.
//
// Revision 1.8  2000/05/31 20:28:08  guilkey
// Added strain_energy_label to the .h file.
//
// Revision 1.7  2000/05/18 18:48:31  jas
// Added gravity.  It is read in from the input file.
//
// Revision 1.6  2000/05/02 17:54:32  sparker
// Implemented more of SerialMPM
//
// Revision 1.5  2000/04/28 20:24:44  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
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

