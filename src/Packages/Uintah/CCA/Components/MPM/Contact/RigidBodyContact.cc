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

void RigidBodyContact::initializeContact(const Patch* /*patch*/,
					 int /*dwindex*/,
					 DataWarehouse* /*new_dw*/)
{

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
    StaticArray<NCVariable<double> > gmass(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity(numMatls),gvelocityME(numMatls);
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],    lb->gMassLabel,     dwi, patch,Ghost::None,0);
      new_dw->get(gvelocity[m],lb->gVelocityLabel, dwi, patch,Ghost::None,0);
    }

#if 0
    if(d_sharedState->getElapsedTime() >= d_stop_time){
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	gvelocity[0][*iter]   = zero;
      }
    }
#endif

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      centerOfMassMom=Vector(0.,0.,0.);
      centerOfMassMass=0.0; 
      for(int n = 0; n < numMatls; n++){
        centerOfMassMom+=gvelocity[n][*iter] * gmass[n][*iter];
        centerOfMassMass+=gmass[n][*iter]; 
      }

      // Set each field's velocity equal to the velocity of material 0
      if(!compare(gmass[0][*iter],0.0)){
        for(int n = 1; n < numMatls; n++){
	  gvelocity[n][*iter].z( gvelocity[0][*iter].z() );
        }
      }
    }

    // Store new velocities in DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwindex = matls->get(m);
      new_dw->modify(gvelocity[m], lb->gVelocityLabel, dwindex, patch);
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
    StaticArray<NCVariable<double> > gmass(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> > gacceleration(numMatls);
    StaticArray<NCVariable<double> > frictionalWork(numMatls);

    for(int m=0;m<matls->size();m++){
      int dwindex = matls->get(m);
      new_dw->get(gmass[m], lb->gMassLabel,dwindex ,patch, Ghost::None, 0);
      new_dw->get(gvelocity_star[m], lb->gVelocityStarLabel, dwindex,
		  patch, Ghost::None, 0);
      new_dw->get(gacceleration[m],  lb->gAccelerationLabel, dwindex,
		  patch, Ghost::None, 0);
      new_dw->allocate(frictionalWork[m], lb->frictionalWorkLabel,
                                                            dwindex, patch);
      frictionalWork[m].initialize(0.);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

    if(d_sharedState->getElapsedTime() >= d_stop_time){
      Vector zero(0.,0.,0.);
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	  gacceleration[0][*iter]    = (zero - gvelocity_star[0][*iter])/delT;
	  gvelocity_star[0][*iter]   = zero;
      }
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      centerOfMassMom=Vector(0.,0.,0.);
      centerOfMassMass=0.0; 
      for(int  n = 0; n < numMatls; n++){
         centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
         centerOfMassMass+=gmass[n][*iter]; 
      }

      // Set each field's velocity equal to the center of mass velocity
      // and adjust the acceleration of each field to account for this
      if(!compare(gmass[0][*iter],0.0)){  // Non-rigid matl
        for(int  n = 1; n < numMatls; n++){
	  Dvdt = Vector(0., 0.,
		  -(gvelocity_star[n][*iter].z() 
		  - gvelocity_star[0][*iter].z())/delT);
	  gvelocity_star[n][*iter].z( gvelocity_star[0][*iter].z() );
	  gacceleration[n][*iter]+=Dvdt;
        }
      }
    }

    // Store new velocities and accelerations in DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->modify(gvelocity_star[m],lb->gVelocityStarLabel, dwi,patch);
      new_dw->modify(gacceleration[m], lb->gAccelerationLabel, dwi,patch);
      new_dw->put(frictionalWork[m],   lb->frictionalWorkLabel,dwi,patch);
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
  t->requires(Task::NewDW, lb->gMassLabel,              Ghost::None);

  t->modifies(             lb->gVelocityStarLabel, mss);
  t->modifies(             lb->gAccelerationLabel, mss);
  t->computes(             lb->frictionalWorkLabel);
}
