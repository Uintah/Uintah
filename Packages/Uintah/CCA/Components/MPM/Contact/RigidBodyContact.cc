// RigidBodyContact.cc

#include "RigidBodyContact.h"
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Util/NotFinished.h>

#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

RigidBodyContact::RigidBodyContact(ProblemSpecP& ps, 
				    SimulationStateP& d_sS)
{
  // Constructor

  IntVector v_f;
  ps->require("vel_fields",v_f);
  ps->require("stop_time",d_stop_time);
  std::cout << "vel_fields = " << v_f << endl;
  
  d_sharedState = d_sS;
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
#if 0
  Vector zero(0.0,0.0,0.0);
  // Unused varible - Steve
  //Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMPMMatls();
  double elapsed_time = d_sharedState->getElapsedTime();

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity(numMatls),gvelocityME(numMatls);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->get(gmass[m],     lb->gMassLabel,     dwindex, patch,Ghost::None,0);
    new_dw->get(gvelocity[m], lb->gVelocityLabel, dwindex, patch,Ghost::None,0);

  }

#if 0
  if(d_sharedState->getElapsedTime() >= d_stop_time){
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	gvelocity[0][*iter]   = zero;
    }
  }
#endif

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
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
  for(int n=0; n< numMatls; n++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->put(gvelocity[n], lb->gMomExedVelocityLabel, dwindex, patch);
  }
#else
  NOT_FINISHED("New task stuff");
#endif
}

void RigidBodyContact::exMomIntegrated(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw)
{
#if 0
  Vector zero(0.0,0.0,0.0);
  // Unused variable - Steve
  // Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  Vector Dvdt(0.0,0.0,0.0);
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMPMMatls();

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity_star(numMatls);
  vector<NCVariable<Vector> > gvelocity_starME(numMatls);
  vector<NCVariable<Vector> > gacceleration(numMatls);
  vector<NCVariable<Vector> > gaccelerationME(numMatls);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->get(gmass[m],lb->gMassLabel,dwindex ,patch, Ghost::None, 0);
    new_dw->get(gvelocity_star[m], lb->gVelocityStarLabel, dwindex,
		  patch, Ghost::None, 0);
    new_dw->get(gacceleration[m], lb->gAccelerationLabel, dwindex,
		  patch, Ghost::None, 0);
  }

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  if(d_sharedState->getElapsedTime() >= d_stop_time){
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	gacceleration[0][*iter]    = (zero - gvelocity_star[0][*iter])/delT;
	gvelocity_star[0][*iter]   = zero;
    }
  }

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int  n = 0; n < numMatls; n++){
       centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
       centerOfMassMass+=gmass[n][*iter]; 
    }

    // Set each field's velocity equal to the center of mass velocity
    // and adjust the acceleration of each field to account for this
    if(!compare(gmass[0][*iter],0.0)){  // Non-rigid matl
      for(int  n = 1; n < numMatls; n++){
	Dvdt = zero;
	Dvdt.z( -(gvelocity_star[n][*iter].z() 
		 - gvelocity_star[0][*iter].z())/delT);
	gvelocity_star[n][*iter].z( gvelocity_star[0][*iter].z() );
	gacceleration[n][*iter]+=Dvdt;
       
      }
    }
  }

  // Store new velocities and accelerations in DataWarehouse
  for(int n = 0; n < numMatls; n++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->put(gvelocity_star[n],lb->gMomExedVelocityStarLabel,dwindex,patch);
    new_dw->put(gacceleration[n], lb->gMomExedAccelerationLabel,dwindex,patch);
  }
#else
  NOT_FINISHED("new task stuff");
#endif
}

void RigidBodyContact::addComputesAndRequiresInterpolated( Task* t,
					     const PatchSet* patches,
					     const MaterialSet* matls) const
{
#if 0
  int idx = matl->getDWIndex();
  t->requires( new_dw, lb->gMassLabel,     idx, patch, Ghost::None);
  t->requires( new_dw, lb->gVelocityLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityLabel, idx, patch );
#else
   NOT_FINISHED("new task stuff");
#endif

}

void RigidBodyContact::addComputesAndRequiresIntegrated( Task* t,
					     const PatchSet* patches,
					     const MaterialSet* matls) const
{
#if 0
  int idx = matl->getDWIndex();
  t->requires(new_dw, lb->gMassLabel,         idx, patch, Ghost::None);
  t->requires(new_dw, lb->gVelocityStarLabel, idx, patch, Ghost::None);
  t->requires(new_dw, lb->gAccelerationLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityStarLabel, idx, patch);
  t->computes( new_dw, lb->gMomExedAccelerationLabel, idx, patch);
#else
   NOT_FINISHED("new task stuff");
#endif

}
