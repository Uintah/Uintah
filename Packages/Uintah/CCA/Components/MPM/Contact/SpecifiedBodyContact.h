// SpecifiedBody.h

#ifndef __SPECIFIED_BODY_H_
#define __SPECIFIED_BODY_H_

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Task.h>

namespace Uintah {

class VarLabel;

/**************************************

CLASS
   SpecifiedBodyContact
   
   Short description...

GENERAL INFORMATION

   SpecifiedBodyContact.h

   Andrew Brydon 
   andrew@lanl.gov

   based on RigidBodyContact.
     Steven G. Parker
     Department of Computer Science
     University of Utah

     Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
     Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_specified_velocity

DESCRIPTION
  One of the derived Contact classes.  Allow motion to of contact body 
  to be specified by by an input file.

  the format of the input is 
  <contact>
    <format>specified</format>
    <filename>fname.txt</filename>
    <direction>[1,1,1]</direction>
  </contact>

  where filename points to an existing test file (which much exist from all 
  mpi processors), storing a list of text columns
     {simtime}  {uvel} {vvel} {wvel}

  the times must be given in ascending order.
  linear interpolation is performed on values, and the end values are used if out
  of the timestep range. 

  the direction can be used to limit the directions which the rigid region affects
  the velocities of the other material. This can be used to impose a normal velocity
  with slip in the other directions.    The default of [1,1,1] applies sticky contact.
  
WARNING
  
****************************************/

      class SpecifiedBodyContact : public Contact {
      private:
	 
	 // Prevent copying of this class
	 // copy constructor
	 SpecifiedBodyContact(const SpecifiedBodyContact &con);
	 SpecifiedBodyContact& operator=(const SpecifiedBodyContact &con);
	 
      private:
         Vector findVel(double t) const;
         
      private:
	 SimulationStateP d_sharedState;
         IntVector d_direction;
         std::vector< std::pair<double, Vector> > d_vel_profile;
	 
      public:
	 // Constructor
	 SpecifiedBodyContact(ProblemSpecP& ps,SimulationStateP& d_sS,MPMLabel* lb,MPMFlags*flag);
	 
	 // Destructor
	 virtual ~SpecifiedBodyContact();

	 // Basic contact methods
	 virtual void exMomInterpolated(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset* matls,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw);
	 
	 virtual void exMomIntegrated(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw);

         virtual void addComputesAndRequiresInterpolated(Task* task,
					     const PatchSet* patches,
					     const MaterialSet* matls) const;

         virtual void addComputesAndRequiresIntegrated(Task* task,
					     const PatchSet* patches,
					     const MaterialSet* matls) const;
      };
      
} // end namespace Uintah

#endif /* __SPECIFIED_BODY_H_ */
