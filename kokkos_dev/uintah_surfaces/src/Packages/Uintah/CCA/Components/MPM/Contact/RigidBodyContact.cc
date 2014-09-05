// RigidBodyContact.cc
#include <Packages/Uintah/CCA/Components/MPM/Contact/RigidBodyContact.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Containers/StaticArray.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>
using std::cerr;

using namespace std;
using namespace Uintah;
using namespace SCIRun;

RigidBodyContact::RigidBodyContact(const ProcessorGroup* myworld,
                                   ProblemSpecP& ps,SimulationStateP& d_sS, 
				   MPMLabel* Mlb, MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_stop_time = 999999.99;  // default is to never stop
  ps->get("stop_time",d_stop_time);

  IntVector defaultDir(0,0,1);
  ps->getWithDefault("direction",d_direction, defaultDir);

  Vector defaultStopVel(0.,0.,0.);
  ps->getWithDefault("velocity_after_stop",d_vel_after_stop, defaultStopVel);

  d_direction.x(1^d_direction.x());  // Change 1 to 0, or 0 to 1
  d_direction.y(1^d_direction.y());  // Change 1 to 0, or 0 to 1
  d_direction.z(1^d_direction.z());  // Change 1 to 0, or 0 to 1
  if (d_direction.x() < 0 || d_direction.x() > 1 || d_direction.y() < 0 ||
      d_direction.y() > 1 || d_direction.z() < 0 || d_direction.z() > 1) {
    throw ProblemSetupException(" E R R O R----->MPM:Dir. of rigid contact should be 0 or 1");
  }
  //cout << "Direction of contact = " << d_direction << endl;
  
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
  
  d_matls.add(0); // always need material 0 for rigid contact
}

RigidBodyContact::~RigidBodyContact()
{
  // Destructor

}

void RigidBodyContact::exMomInterpolated(const ProcessorGroup*,
					 const PatchSubset* patches,
					 const MaterialSubset* matls,
					 DataWarehouse*,
					 DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Retrieve necessary data from DataWarehouse
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity(numMatls);
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],    lb->gMassLabel,     dwi, patch,Ghost::None,0);
      new_dw->getModifiable(gvelocity[m],lb->gVelocityLabel, dwi, patch);
    }

#if 0
    if(d_sharedState->getElapsedTime() >= d_stop_time){
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter; 
	gvelocity[0][c]   = d_vel_after_stop;
      }
    }
#endif

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      // Set each field's velocity equal to the velocity of material 0
      if(d_matls.present(gmass, *iter)) {
        for(int n = 1; n < numMatls; n++){
          int xn = d_direction.x()*n;
          int yn = d_direction.y()*n;
          int zn = d_direction.z()*n;
          // set each velocity component either to it's own velocity
          // or that of the rigid body
	  gvelocity[n][c].x( gvelocity[xn][c].x() );
	  gvelocity[n][c].y( gvelocity[yn][c].y() );
	  gvelocity[n][c].z( gvelocity[zn][c].z() );
        }
      }
    }

  }
}

void RigidBodyContact::exMomIntegrated(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw)
{
  Vector Dvdt(0.0,0.0,0.0);

  int numMatls = d_sharedState->getNumMPMMatls();
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Retrieve necessary data from DataWarehouse
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<Vector> > gvelocity(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> > gacceleration(numMatls);
    StaticArray<NCVariable<double> > frictionWork(numMatls);

    for(int m=0;m<matls->size();m++){
     int dwi = matls->get(m);
     new_dw->get(gmass[m],    lb->gMassLabel,          dwi,patch,Ghost::None,0);
     new_dw->get(gvelocity[m],lb->gVelocityInterpLabel,dwi,patch,Ghost::None,0);
     new_dw->getModifiable(gvelocity_star[m],lb->gVelocityStarLabel, dwi,patch);
     new_dw->getModifiable(gacceleration[m], lb->gAccelerationLabel, dwi,patch);
     new_dw->getModifiable(frictionWork[m],lb->frictionalWorkLabel,dwi,patch);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    static bool stopped = false;
    Vector new_velocity = d_vel_after_stop;
    if(d_sharedState->getElapsedTime() >= d_stop_time && !stopped){
      stopped = true;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
          IntVector c = *iter; 
          gacceleration[0][c]    = (new_velocity - gvelocity[0][c])/delT;
          gvelocity_star[0][c]   = new_velocity;
      }
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      if(d_matls.present(gmass, *iter)) {
        for(int  n = 1; n < numMatls; n++){
          int xn = d_direction.x()*n;
          int yn = d_direction.y()*n;
          int zn = d_direction.z()*n;
          double xDvdt = (gvelocity_star[xn][c].x() - gvelocity[n][c].x())/delT;
          double yDvdt = (gvelocity_star[yn][c].y() - gvelocity[n][c].y())/delT;
          double zDvdt = (gvelocity_star[zn][c].z() - gvelocity[n][c].z())/delT;
          Dvdt = Vector(xDvdt, yDvdt, zDvdt);
          gvelocity_star[n][c].x( gvelocity_star[xn][c].x() );
          gvelocity_star[n][c].y( gvelocity_star[yn][c].y() );
          gvelocity_star[n][c].z( gvelocity_star[zn][c].z() );
          gacceleration[n][c]=Dvdt;
        }
      }
    }
  }
}

void RigidBodyContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* ms) 
{
  Task * t = new Task("RigidBodyContact::exMomInterpolated", 
                      this, &RigidBodyContact::exMomInterpolated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->modifies(             lb->gVelocityLabel, mss);
  
  sched->addTask(t, patches, ms);
}

void RigidBodyContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* ms) 
{
  Task * t = scinew Task("RigidBodyContact::exMomIntegrated", 
                         this, &RigidBodyContact::exMomIntegrated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);    
  t->requires(Task::NewDW, lb->gMassLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityInterpLabel, Ghost::None);
  t->modifies(             lb->gVelocityStarLabel, mss);
  t->modifies(             lb->gAccelerationLabel, mss);
  t->modifies(             lb->frictionalWorkLabel, mss);

 //__________________________________
 //  add requirements for Northrup Grumman nozzle
 #define rigidBody_1
 #include "../../MPMICE/NGC_nozzle.i"
 #undef rigidBody_1

  
  sched->addTask(t, patches, ms);
}


