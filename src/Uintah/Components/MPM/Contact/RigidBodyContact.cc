//
// $Id$
//

// RigidBodyContact.cc
//

#include "RigidBodyContact.h"
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/VarLabel.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <Uintah/Components/MPM/MPMLabel.h>

using namespace std;
using namespace Uintah::MPM;
using SCICore::Geometry::Vector;
using SCICore::Geometry::IntVector;
using std::vector;

RigidBodyContact::RigidBodyContact(ProblemSpecP& ps, 
				    SimulationStateP& d_sS)
{
  // Constructor

  IntVector v_f;
  ps->require("vel_fields",v_f);
  std::cout << "vel_fields = " << v_f << endl;
  
  d_sharedState = d_sS;
}

RigidBodyContact::~RigidBodyContact()
{
  // Destructor

}

void RigidBodyContact::initializeContact(const Patch* /*patch*/,
					 int /*dwindex*/,
					 DataWarehouseP& /*new_dw*/)
{

}

void RigidBodyContact::exMomInterpolated(const ProcessorGroup*,
					 const Patch* patch,
					 DataWarehouseP&,
					 DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMPMMatls();

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity(numMatls);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->get(gmass[m], lb->gMassLabel,dwindex , patch,
		  Ghost::None, 0);
    new_dw->get(gvelocity[m], lb->gVelocityLabel, dwindex, patch,
		  Ghost::None, 0);
  }

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int n = 0; n < numMatls; n++){
      centerOfMassMom+=gvelocity[n][*iter] * gmass[n][*iter];
      centerOfMassMass+=gmass[n][*iter]; 
    }

    // Set each field's velocity equal to the velocity of material 0
    if(!compare(gmass[1][*iter],0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int n = 1; n < numMatls; n++){
       if(!compare(gvelocity[n][*iter].x(),centerOfMassVelocity.x()) ||
	  !compare(gvelocity[n][*iter].y(),centerOfMassVelocity.y()) ||	
	  !compare(gvelocity[n][*iter].z(),centerOfMassVelocity.z()) ){	
//           cout << "gvn = " << gvelocity[n][*iter] << " " << n << endl;
//           cout << "cmv = " << centerOfMassVelocity << endl;
//           cout << "gmass = " << gmass[n][*iter] << " " << n << endl;
//           cout << "cmmmm = " << centerOfMassMass << endl;
	    gvelocity[n][*iter] = gvelocity[0][*iter];
	}
      }
    }
  }

  // Store new velocities in DataWarehouse
  for(int n=0; n< numMatls; n++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->put(gvelocity[n], lb->gMomExedVelocityLabel, dwindex, patch);
  }
}

void RigidBodyContact::exMomIntegrated(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  Vector Dvdt;
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMPMMatls();

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity_star(numMatls);
  vector<NCVariable<Vector> > gacceleration(numMatls);
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

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int  n = 0; n < numMatls; n++){
       centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
       centerOfMassMass+=gmass[n][*iter]; 
    }

    // Set each field's velocity equal to the center of mass velocity
    // and adjust the acceleration of each field to account for this
    if(!compare(gmass[1][*iter],0.0)){  // Non-rigid matl
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int  n = 1; n < numMatls; n++){
       if(!compare(gvelocity_star[n][*iter].x(),centerOfMassVelocity.x()) ||
	  !compare(gvelocity_star[n][*iter].y(),centerOfMassVelocity.y()) ||	
	  !compare(gvelocity_star[n][*iter].z(),centerOfMassVelocity.z()) ){
//           cout << "vstarn =  " << gvelocity_star[n][*iter] << endl;
//           cout << "cmvstar = " << centerOfMassVelocity << endl;
	    Dvdt = -(gvelocity_star[n][*iter] - gvelocity_star[0][*iter])/delT;
	    gvelocity_star[n][*iter] = gvelocity_star[0][*iter];
	    gacceleration[n][*iter]+=Dvdt;
       }
      }
    }
  }

  // Store new velocities and accelerations in DataWarehouse
  for(int n = 0; n < numMatls; n++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->put(gvelocity_star[n], lb->gMomExedVelocityStarLabel,dwindex,patch);
    new_dw->put(gacceleration[n],  lb->gMomExedAccelerationLabel,dwindex,patch);
  }
}

void RigidBodyContact::addComputesAndRequiresInterpolated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{
  int idx = matl->getDWIndex();
  t->requires( new_dw, lb->gMassLabel,     idx, patch, Ghost::None);
  t->requires( new_dw, lb->gVelocityLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityLabel, idx, patch );

}

void RigidBodyContact::addComputesAndRequiresIntegrated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{

  int idx = matl->getDWIndex();
  t->requires(new_dw, lb->gMassLabel,         idx, patch, Ghost::None);
  t->requires(new_dw, lb->gVelocityStarLabel, idx, patch, Ghost::None);
  t->requires(new_dw, lb->gAccelerationLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityStarLabel, idx, patch);
  t->computes( new_dw, lb->gMomExedAccelerationLabel, idx, patch);

}

// $Log$
// Revision 1.1  2001/01/11 03:31:31  guilkey
// Created new contact model for rigid bodies.
//
