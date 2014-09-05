
// SingleVel.cc
// One of the derived Contact classes.  This particular
// class contains methods for recapturing single velocity
// field behavior from objects belonging to multiple velocity
// fields.  The main purpose of this type of contact is to
// ensure that one can get the same answer using prescribed
// contact as can be gotten using "automatic" contact.
#include <Packages/Uintah/CCA/Components/MPM/Contact/SingleVelContact.h>
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
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Containers/StaticArray.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
using std::vector;

SingleVelContact::SingleVelContact(const ProcessorGroup* myworld,
                                   ProblemSpecP& ps, SimulationStateP& d_sS, 
				   MPMLabel* Mlb,MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
}

SingleVelContact::~SingleVelContact()
{
  // Destructor

}

void SingleVelContact::exMomInterpolated(const ProcessorGroup*,
					 const PatchSubset* patches,
					 const MaterialSubset* matls,
					 DataWarehouse*,
					 DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector centerOfMassVelocity(0.0,0.0,0.0);

    // Retrieve necessary data from DataWarehouse
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity(numMatls);
    for(int m=0;m<matls->size();m++){
      int dwindex = matls->get(m);
      new_dw->get(gmass[m], lb->gMassLabel,    dwindex, patch,Ghost::None,0);
      new_dw->getModifiable(gvelocity[m], lb->gVelocityLabel,dwindex, patch);
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      if(d_matls.present(gmass, c)) {
        
        Vector centerOfMassMom(0,0,0);
        double centerOfMassMass=0.0;
        
        for(int n = 0; n < numMatls; n++){
          if(d_matls.requested(n)) {
            centerOfMassMom+=gvelocity[n][c] * gmass[n][c];
            centerOfMassMass+=gmass[n][c]; 
          }
        }
        
        // Set each field's velocity equal to the center of mass velocity
        centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
        for(int n = 0; n < numMatls; n++) {
          if(d_matls.requested(n)) {
            gvelocity[n][c] = centerOfMassVelocity;
          }
        }

      }
      
    }
  }
}

void SingleVelContact::exMomIntegrated(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector zero(0.0,0.0,0.0);
    Vector centerOfMassVelocity(0.0,0.0,0.0);
    Vector centerOfMassMom(0.0,0.0,0.0);
    Vector Dvdt;
    double centerOfMassMass;

    // Retrieve necessary data from DataWarehouse
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> > gacceleration(numMatls);
    StaticArray<NCVariable<double> > frictionWork(numMatls);

    for(int m=0;m<matls->size();m++){
     int dwi = matls->get(m);
     new_dw->get(gmass[m],lb->gMassLabel, dwi, patch, Ghost::None, 0);
     new_dw->getModifiable(gvelocity_star[m],lb->gVelocityStarLabel, dwi,patch);
     new_dw->getModifiable(gacceleration[m], lb->gAccelerationLabel, dwi,patch);
     new_dw->getModifiable(frictionWork[m], lb->frictionalWorkLabel, dwi,
                           patch);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      if(d_matls.present(gmass, c)) {

        centerOfMassMom=zero;
        centerOfMassMass=0.0; 
        for(int  n = 0; n < numMatls; n++){
          if(d_matls.requested(n)) {
            centerOfMassMom+=gvelocity_star[n][c] * gmass[n][c];
            centerOfMassMass+=gmass[n][c]; 
          }
        }
        
        // Set each field's velocity equal to the center of mass velocity
        // and adjust the acceleration of each field to account for this
        centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
        for(int  n = 0; n < numMatls; n++){
          if(d_matls.requested(n)) {
            Dvdt = (centerOfMassVelocity - gvelocity_star[n][c])/delT;
            gvelocity_star[n][c] = centerOfMassVelocity;
            gacceleration[n][c]+=Dvdt;
          }
        }
      }
    }
    
  }
}

void SingleVelContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
						  const PatchSet* patches,
				     		  const MaterialSet* ms)
{
  Task * t = new Task("SingleVelContact::exMomInterpolated", 
                      this, &SingleVelContact::exMomInterpolated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires( Task::NewDW, lb->gMassLabel,          Ghost::None);

  t->modifies(              lb->gVelocityLabel, mss);
  
  sched->addTask(t, patches, ms);
}

void SingleVelContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* ms) 
{
  Task * t = new Task("SingleVelContact::exMomIntegrated", 
                      this, &SingleVelContact::exMomIntegrated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);    
  t->requires(Task::NewDW, lb->gMassLabel,              Ghost::None);

  t->modifies(             lb->gVelocityStarLabel, mss);
  t->modifies(             lb->gAccelerationLabel, mss);
  t->modifies(             lb->frictionalWorkLabel,mss);
  
  sched->addTask(t, patches, ms);
}
