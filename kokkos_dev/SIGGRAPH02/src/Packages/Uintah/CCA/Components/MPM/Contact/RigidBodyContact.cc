// RigidBodyContact.cc

#include "RigidBodyContact.h"
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/NotFinished.h>

#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

RigidBodyContact::RigidBodyContact(ProblemSpecP& ps, 
				    SimulationStateP& d_sS, MPMLabel* Mlb)
{
  // Constructor

  IntVector v_f;
  ps->require("vel_fields",v_f);
  ps->get("stop_time",d_stop_time);
  
  d_sharedState = d_sS;
  lb = Mlb;
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
  //  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  double centerOfMassMass;

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
	gvelocity[0][c]   = zero;
      }
    }
#endif

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      centerOfMassMom=Vector(0.,0.,0.);
      centerOfMassMass=0.0; 
      for(int n = 0; n < numMatls; n++){
        centerOfMassMom+=gvelocity[n][c] * gmass[n][c];
        centerOfMassMass+=gmass[n][c]; 
      }

      // Set each field's velocity equal to the velocity of material 0
      if(!compare(gmass[0][c],0.0)){
        for(int n = 1; n < numMatls; n++){
	  gvelocity[n][c].z( gvelocity[0][c].z() );
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
  Vector centerOfMassMom(0.0,0.0,0.0);
  Vector Dvdt(0.0,0.0,0.0);
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMPMMatls();
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Retrieve necessary data from DataWarehouse
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> > gacceleration(numMatls);
    StaticArray<NCVariable<double> > frictionWork(numMatls);

    for(int m=0;m<matls->size();m++){
     int dwi = matls->get(m);
     new_dw->get(gmass[m], lb->gMassLabel,dwi ,patch, Ghost::None, 0);
     new_dw->getModifiable(gvelocity_star[m],lb->gVelocityStarLabel, dwi,patch);
     new_dw->getModifiable(gacceleration[m], lb->gAccelerationLabel, dwi,patch);
     new_dw->allocateAndPut(frictionWork[m], lb->frictionalWorkLabel,dwi,patch);
     frictionWork[m].initialize(0.);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

    if(d_sharedState->getElapsedTime() >= d_stop_time){
      Vector zero(0.,0.,0.);
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
          IntVector c = *iter; 
	  gacceleration[0][c]    = (zero - gvelocity_star[0][c])/delT;
	  gvelocity_star[0][c]   = zero;
      }
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      centerOfMassMom=Vector(0.,0.,0.);
      centerOfMassMass=0.0; 
      for(int  n = 0; n < numMatls; n++){
         centerOfMassMom+=gvelocity_star[n][c] * gmass[n][c];
         centerOfMassMass+=gmass[n][c]; 
      }

      // Set each field's velocity equal to the center of mass velocity
      // and adjust the acceleration of each field to account for this
      if(!compare(gmass[0][c],0.0)){  // Non-rigid matl
        for(int  n = 1; n < numMatls; n++){
	  Dvdt = Vector(0., 0.,
		  -(gvelocity_star[n][c].z() 
		  - gvelocity_star[0][c].z())/delT);
	  gvelocity_star[n][c].z( gvelocity_star[0][c].z() );
	  gacceleration[n][c]+=Dvdt;
        }
      }
    }
  }
}

void RigidBodyContact::addComputesAndRequiresInterpolated( Task* t,
					     const PatchSet* ,
					     const MaterialSet* ms) const
{
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);

  t->modifies(             lb->gVelocityLabel, mss);
}

void RigidBodyContact::addComputesAndRequiresIntegrated( Task* t,
					     const PatchSet* ,
					     const MaterialSet* ms) const
{
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);    
  t->requires(Task::NewDW, lb->gMassLabel,              Ghost::None);

  t->modifies(             lb->gVelocityStarLabel, mss);
  t->modifies(             lb->gAccelerationLabel, mss);
  t->computes(             lb->frictionalWorkLabel);
}
