// SpecifiedBody.h

#ifndef __SPECIFIED_BODY_H_
#define __SPECIFIED_BODY_H_

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactMaterialSpec.h>
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
  One of the derived Contact classes.  Allow motion of a body 
  to be specified by an input file.

  the format of the input is 
  <contact>
    <type>specified</type>
    <filename>fname.txt</filename>
    <direction>[1,1,1]</direction>
    <material>0</material> 
  </contact>

  where filename points to an existing test file (which much exist from all 
  mpi processors), storing a list of text columns
     {simtime}  {uvel} {vvel} {wvel}

  the times must be given in ascending order.
  linear interpolation is performed on values, and the end values are used if out
  of the time range. 

  the direction can be used to limit the directions which the rigid region affects
  the velocities of the other material. This can be used to impose a normal velocity
  with slip in the other directions. The default of [1,1,1] applies sticky contact.
  
  the material is the rigid material to use, and is optional. default is 0.
  
  There are two alternate formats (which exist for compatability with RigidBodyContact)

  <contact>
    <direction>[0,0,1]</direction>
  </contact>

  Apply center-of-mass velocity to objects in contact with material (default of 0 assumed).
  

  <contact>
    <direction>[1,1,1]</direction>
    <stop_time>2.0</stop_time>
    <velocity_after_stop>[0,1,0]</velocity_after_stop>
  <contact>

  when t>stop_time, impose velocity_after_stop.
  This can be combined with either rigid velocity (as shown) or a velocity profile to 
  produce wild and wacky results. 

  the velocity_after_stop is optional, with default (0,0,0).

  
  Notes: (Probably out of date - Jim ??)
     
     Contact conditions are specified in two stages, and deal with 4 velocity fields
     
     v^k        velocity at start of exMomInterpolated
     v*^k       velocity coming out of exMomInterpolated (with rigid cells set)
     v^k+1      velocity coming in to exMomIntegrated
     v*^k+1     velocity coming out of exMomIntegrated (with rigid cells set)

     a^k+1/2    acceleration
     a*^k+1/2   acceleration in rigid cells

     for non-contact cells, we expect
        a^k+1/2 = ( v^k+1 - v^k)/dt   (basic center diff)     [1]

     for rigid cells, we require the acceleration
        a*^k+1/2 = (v*k+1 - v^k)/dt                           [2]
     
     
     // this option is not done, v^k is stored throughout ....
     //
     to avoid having to store v^k through the entire iteration, we store
        a*^k = (v*^k - v^k)/dt                                [3]
     in exMomInterpolated
     
     and then
        a*^k+1/2 = (v*k+1/2 - v*^k)/dt + a*^k                 [4]
     in exMomIntegrated, which reduces to [2] as required.
     
  ****************************************/
  
  class SpecifiedBodyContact : public Contact {
  private:
	 
    // Prevent copying of this class
    // copy constructor
    SpecifiedBodyContact(const SpecifiedBodyContact &con);
    SpecifiedBodyContact& operator=(const SpecifiedBodyContact &con);
	 
  private:
    Vector findVelFromProfile(double t) const;
    
  private:
    SimulationStateP d_sharedState;
    double    d_stop_time;
    Vector    d_vel_after_stop;
    int       d_material;
    IntVector d_direction;
    std::vector< std::pair<double, Vector> > d_vel_profile;
    
  public:
    // Constructor
    SpecifiedBodyContact(const ProcessorGroup* myworld,
                         ProblemSpecP& ps,SimulationStateP& d_sS,MPMLabel* lb,MPMFlags*flag);
	 
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

    virtual void addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                    const PatchSet* patches,
                                                    const MaterialSet* matls);

    virtual void addComputesAndRequiresIntegrated(SchedulerP & sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls);
  };
      
} // end namespace Uintah

#endif /* __SPECIFIED_BODY_H_ */
