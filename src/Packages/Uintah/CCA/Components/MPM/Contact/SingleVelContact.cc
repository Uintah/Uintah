
// SingleVel.cc
// One of the derived Contact classes.  This particular
// class contains methods for recapturing single velocity
// field behavior from objects belonging to multiple velocity
// fields.  The main purpose of this type of contact is to
// ensure that one can get the same answer using prescribed
// contact as can be gotten using "automatic" contact.

#include "SingleVelContact.h"
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
//#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Util/NotFinished.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
using std::vector;

SingleVelContact::SingleVelContact(ProblemSpecP& ps, 
				    SimulationStateP& d_sS)
{
  // Constructor

  IntVector v_f;
  ps->require("vel_fields",v_f);
  std::cout << "vel_fields = " << v_f << endl;
  
  d_sharedState = d_sS;
}

SingleVelContact::~SingleVelContact()
{
  // Destructor

}

void SingleVelContact::initializeContact(const Patch* /*patch*/,
					 int /*dwindex*/,
					 DataWarehouse* /*new_dw*/)
{

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
    vector<NCVariable<double> > gmass(numMatls);
    vector<NCVariable<Vector> > gvelocity(numMatls);
    for(int m=0,idx=0;m<matls->size();m++,idx++){
      int dwindex = matls->get(m);
      new_dw->get(gmass[idx], lb->gMassLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(gvelocity[idx], lb->gVelocityLabel, dwindex, patch,
		  Ghost::None, 0);
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      Vector centerOfMassMom(0,0,0);
      double centerOfMassMass=0.0;
      for(int n = 0; n < numMatls; n++){
	centerOfMassMom+=gvelocity[n][*iter] * gmass[n][*iter];
	centerOfMassMass+=gmass[n][*iter]; 
      }

      // Set each field's velocity equal to the center of mass velocity
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int n = 0; n < numMatls; n++){
	gvelocity[n][*iter] = centerOfMassVelocity;
      }
    }

    // Store new velocities in DataWarehouse
    for(int m=0,idx=0;m<matls->size();m++,idx++){
      int dwindex = matls->get(m);
      new_dw->put(gvelocity[idx], lb->gMomExedVelocityLabel, dwindex, patch);
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
    vector<NCVariable<double> > gmass(numMatls);
    vector<NCVariable<Vector> > gvelocity_star(numMatls);
    vector<NCVariable<Vector> > gacceleration(numMatls);
    for(int m=0,idx=0;m<matls->size();m++,idx++){
      int dwindex = matls->get(m);
      new_dw->get(gmass[idx],lb->gMassLabel, dwindex, patch, Ghost::None, 0);
      new_dw->get(gvelocity_star[idx], lb->gVelocityStarLabel, dwindex,
		  patch, Ghost::None, 0);
      new_dw->get(gacceleration[idx], lb->gAccelerationLabel, dwindex,
		  patch, Ghost::None, 0);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      centerOfMassMom=zero;
      centerOfMassMass=0.0; 
      for(int  n = 0; n < numMatls; n++){
	centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
	centerOfMassMass+=gmass[n][*iter]; 
      }

      // Set each field's velocity equal to the center of mass velocity
      // and adjust the acceleration of each field to account for this
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int  n = 0; n < numMatls; n++){
	Dvdt = (centerOfMassVelocity - gvelocity_star[n][*iter])/delT;
	gvelocity_star[n][*iter] = centerOfMassVelocity;
	gacceleration[n][*iter]+=Dvdt;
      }
    }

    // Store new velocities and accelerations in DataWarehouse
    for(int m=0,idx=0;m<matls->size();m++,idx++){
      int dwindex = matls->get(m);
      new_dw->put(gvelocity_star[idx], lb->gMomExedVelocityStarLabel,
		  dwindex,patch);
      new_dw->put(gacceleration[idx],  lb->gMomExedAccelerationLabel,
		  dwindex,patch);
    }
  }
}

void SingleVelContact::addComputesAndRequiresInterpolated( Task* t,
					     const PatchSet* patches,
					     const MaterialSet* matls) const
{
  t->requires( Task::NewDW, lb->gMassLabel,     Ghost::None);
  t->requires( Task::NewDW, lb->gVelocityLabel, Ghost::None);

  t->computes(lb->gMomExedVelocityLabel);
}

void SingleVelContact::addComputesAndRequiresIntegrated( Task* t,
					     const PatchSet* patches,
					     const MaterialSet* matls) const
{
  t->requires(Task::NewDW, lb->gMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityStarLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gAccelerationLabel, Ghost::None);

  t->computes(lb->gMomExedVelocityStarLabel);
  t->computes(lb->gMomExedAccelerationLabel);
}

