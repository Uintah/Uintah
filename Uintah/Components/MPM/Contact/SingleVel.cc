// SingleVel.cc
//
// One of the derived Contact classes.  This particular
// class contains methods for recapturing single velocity
// field behavior from objects belonging to multiple velocity
// fields.  The main purpose of this type of contact is to
// ensure that one can get the same answer using prescribed
// contact as can be gotten using "automatic" contact.

#include "SingleVel.h"

SingleVel::SingleVel()
{
  // Constructor

}

SingleVel::~SingleVel()
{
  // Destructor

}

void SingleVel::exMomInterpolated(const Region* region,
                                  const DataWarehouseP& old_dw,
                                  DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector CenterOfMassVelocity(0.0,0.0,0.0);
  Vector CenterOfMassMom(0.0,0.0,0.0);
  double centerOfMassMass;
  int n;

  // Retrieve necessary data from DataWarehouse
  for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
    NCVariable<double> gmass[n];
    new_dw->get(gmass[n], "g.mass", n, region, 0);
    NCVariable<Vector> gvelocity[n];
    new_dw->get(gvelocity[n], "g.velocity", n, region, 0);
  }

  for(NodeIterator iter = region->begin();
             iter != region->end(); iter++){
    CenterOfMassMom=zero;
    CenterOfMassMass=0.0; 
    for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
       centerOfMassMom+=gvelocity[n][*iter] * gmass[n][*iter];
       centerOfMassMass+=mass[n][i]; 
    }

    // Set each field's velocity equal to the center of mass velocity
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
	gvelocity[n][*iter] = centerOfMassVelocity;
      }
    }
  }

  // Store new velocities in DataWarehouse
  for( int n=firstMPMVelField; n<=numMPMVelFields; n++){
    new_dw->put(gvelocity[n], "g.velocity", n, region, 0);
  }

}

void SingleVel::exMomIntegrated(const Region* region,
                                const DataWarehouseP& old_dw,
                                DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector CenterOfMassVelocity(0.0,0.0,0.0);
  Vector CenterOfMassMom(0.0,0.0,0.0);
  double centerOfMassMass;
  int n;

  // Retrieve necessary data from DataWarehouse
  for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
    NCVariable<double> gmass[n];
    new_dw->get(gmass[n], "g.mass", n, region, 0);
    NCVariable<Vector> gvelocity_star[n];
    new_dw->get(gvelocity_star[n], "g.velocity_star", n, region, 0);
    NCVariable<Vector> acceleration[n];
    new_dw->get(acceleration[n], "g.acceleration", n, region, 0);
  }
  SoleVariable<double> delt;
  old_dw->get(delt, "delt");


  for(NodeIterator iter = region->begin();
             iter != region->end(); iter++){
    CenterOfMassMom=zero;
    CenterOfMassMass=0.0; 
    for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
       centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
       centerOfMassMass+=mass[n][i]; 
    }

    // Set each field's velocity equal to the center of mass velocity
    // and adjust the acceleration of each field to account for this
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
        Dvdt = (centerOfMassVelocity - velocity_star[n][*iter])/delt;
	gvelocity_star[n][*iter] = centerOfMassVelocity;
	acceleration[n][*iter]+=Dvdt;
      }
    }
  }

  // Store new velocities and accelerations in DataWarehouse
  for( n=firstMPMVelField; n<=numMPMVelFields; n++){
    new_dw->put(gvelocity_star[n], "g.velocity_star", n, region, 0);
    new_dw->put(acceleration[n], "g.acceleration", n, region, 0);
  }

}

// $Log$
// Revision 1.1  2000/03/16 01:05:13  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//
