// SpecifiedBodyContact.cc
#include <Packages/Uintah/CCA/Components/MPM/Contact/SpecifiedBodyContact.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
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

SpecifiedBodyContact::SpecifiedBodyContact(ProblemSpecP& ps,SimulationStateP& d_sS, 
				   MPMLabel* Mlb, MPMFlags* MFlag)
{
  // Constructor
  // read a list of values from a file
  std::string fname;
  ps->require("filename", fname);
  
  IntVector defaultDir(1,1,1);
  ps->getWithDefault("direction",d_direction, defaultDir);
 
  std::ifstream is(fname.c_str());
  if (!is ){
    throw ProblemSetupException("ERROR: opening MPM rigid motion file '"+fname+"'\nFailed to find profile file");
  }
  double t0(-100);
  while(is)
    {
      double t1;
      double vx, vy, vz;
      is >> t1 >> vx >> vy >> vz;
      if(is)
        {
          if(t1<=t0)
            throw ProblemSetupException("ERROR: profile file is not monotomically increasing");
          d_vel_profile.push_back( std::pair<double,Vector>(t1,Vector(vx,vy,vz)) );
        }
      t0 = t1;
    }
  if(d_vel_profile.size()<2)
    {
      throw ProblemSetupException("ERROR: Failed to generate value velocity profile");
    }


  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
}

SpecifiedBodyContact::~SpecifiedBodyContact()
{
  // Destructor

}

Vector
SpecifiedBodyContact::findVel(double t) const
{
  int smin = 0;
  int smax = d_vel_profile.size()-1;
  double tmin = d_vel_profile[0].first;
  double tmax = d_vel_profile[smax].first;
  if(t<=tmin)
    {
      return d_vel_profile[0].second;
    }
  else if(t>=tmax)
    {
      return d_vel_profile[smax].second;
    }
  else
    {

      while (smax>smin+1)
        {
          int smid = (smin+smax)/2;
          if(d_vel_profile[smid].first<t)
            smin = smid;
          else
            smax = smid;
        }
      double l  = (d_vel_profile[smin+1].first-d_vel_profile[smin].first);
      double xi = (t-d_vel_profile[smin].first)/l;
      double vx = xi*d_vel_profile[smin+1].second[0]+(1-xi)*d_vel_profile[smin].second[0];
      double vy = xi*d_vel_profile[smin+1].second[1]+(1-xi)*d_vel_profile[smin].second[1];
      double vz = xi*d_vel_profile[smin+1].second[2]+(1-xi)*d_vel_profile[smin].second[2];
      return Vector(vx,vy,vz);
    }
}



void SpecifiedBodyContact::exMomInterpolated(const ProcessorGroup*,
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
    double tcurr = d_sharedState->getElapsedTime();
    Vector new_velocity = findVel(tcurr);

    // set velocity to profile vel
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      
      if(!compare(gmass[0][c],0.0)){ // only update if shares cell with rigid body
        for(int n = 1; n < numMatls; n++){ // dont update rigid body here
          // set each velocity component that is being modified to new velocity
          if(d_direction[0]) gvelocity[n][c].x( new_velocity.x() );
          if(d_direction[1]) gvelocity[n][c].y( new_velocity.y() );
          if(d_direction[2]) gvelocity[n][c].z( new_velocity.z() );
        }
      }
    }
  }
}

void SpecifiedBodyContact::exMomIntegrated(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw)
{
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
     if (flag->d_fracture)
       new_dw->getModifiable(frictionWork[m],lb->frictionalWorkLabel,dwi,
			     patch);
     else {
       new_dw->allocateAndPut(frictionWork[m], lb->frictionalWorkLabel,dwi,
			      patch);
       frictionWork[m].initialize(0.);
     }
    }
    
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // set velocity to profile vel
    const double tcurr = d_sharedState->getElapsedTime();
    Vector new_velocity = findVel(tcurr);
    
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      
      if(!compare(gmass[0][c],0.0)){  // rigid matl is always index 0
        
        for(int  n = 0; n < numMatls; n++){ // update material 0 to new velocity also.
          Vector new_vel( gvelocity_star[n][c] );
          if(d_direction[0]) new_vel.x( new_velocity.x() );
          if(d_direction[1]) new_vel.y( new_velocity.y() );
          if(d_direction[2]) new_vel.z( new_velocity.z() );
          gacceleration[n][c]   += (new_vel - gvelocity_star[n][c])/delT;
          gvelocity_star[n][c]   = new_vel;
        }
      }
    }
  }
}

void SpecifiedBodyContact::addComputesAndRequiresInterpolated( Task* t,
					     const PatchSet* ,
					     const MaterialSet* ms) const
{
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);

  t->modifies(             lb->gVelocityLabel, mss);
}

void SpecifiedBodyContact::addComputesAndRequiresIntegrated( Task* t,
					     const PatchSet*,
					     const MaterialSet* ms) const
{
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);    
  t->requires(Task::NewDW, lb->gMassLabel,              Ghost::None);
  t->modifies(             lb->gVelocityStarLabel, mss);
  t->modifies(             lb->gAccelerationLabel, mss);
  if (flag->d_fracture)
    t->modifies(             lb->frictionalWorkLabel, mss);
  else
    t->computes(             lb->frictionalWorkLabel);
}


