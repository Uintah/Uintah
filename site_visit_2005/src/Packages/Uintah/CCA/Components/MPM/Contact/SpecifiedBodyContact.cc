// SpecifiedBodyContact.cc
#include <Packages/Uintah/CCA/Components/MPM/Contact/SpecifiedBodyContact.h>
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
#include <limits>
#include <sgi_stl_warnings_on.h>
using std::cerr;

using namespace std;
using namespace Uintah;
using namespace SCIRun;

SpecifiedBodyContact::SpecifiedBodyContact(const ProcessorGroup* myworld,
                                           ProblemSpecP& ps,
                                           SimulationStateP& d_sS, 
                                           MPMLabel* Mlb, MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  // read a list of values from a file
  std::string fname;
  ps->get("filename", fname);
  
  IntVector defaultDir(0,0,1);
  ps->getWithDefault("direction",d_direction, defaultDir);
  
  ps->getWithDefault("master_material", d_material, 0);
  d_matls.add(d_material); // always need specified material
  
  if(fname!="") {
    std::ifstream is(fname.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR: opening MPM rigid motion file '"+fname+"'\nFailed to find profile file");
    }
    double t0(-1.e9);
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
        throw ProblemSetupException("ERROR: Failed to generate valid velocity profile");
      }
  }
  
  // disable all changes after this time
  ps->getWithDefault("stop_time",d_stop_time, std::numeric_limits<double>::max());
  ps->getWithDefault("velocity_after_stop",d_vel_after_stop, Vector(0,0,0));
  
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
}

SpecifiedBodyContact::~SpecifiedBodyContact()
{
}

// find velocity from table of values
Vector
SpecifiedBodyContact::findVelFromProfile(double t) const
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
      // bisection search on table
      // could probably speed this up by keeping copy of last successful
      // search, and looking at that point and a couple to the right
      //
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

// apply boundary conditions to interpolated velocity v^k
void SpecifiedBodyContact::exMomInterpolated(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* matls,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // Retrieve necessary data from DataWarehouse
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<     NCVariable<Vector> > gvelocity(numMatls);
    StaticArray<     NCVariable<double> > frictionWork(numMatls);
    
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],           lb->gMassLabel,     dwi, patch,Ghost::None,0);
      new_dw->getModifiable(gvelocity[m],lb->gVelocityLabel,     dwi, patch);
      new_dw->getModifiable(frictionWork[m],lb->frictionalWorkLabel,dwi,patch);
    }
    const double tcurr = d_sharedState->getElapsedTime();
    
    // three ways to get velocity 
    //   if > stop time, always use stop velocity
    //   if we have a specified profile, use value from the velocity profile
    //   otherwise, apply rigid velocity to all cells that share a rigid body.
    //
    
    bool  rigid_velocity = true;
    Vector requested_velocity;
    if(tcurr>d_stop_time) {
      requested_velocity = d_vel_after_stop;
      rigid_velocity = false;
    } else if(d_vel_profile.size()>0) {
      requested_velocity = findVelFromProfile(tcurr);
      rigid_velocity  = false;
    }
    
    // Set each field's velocity equal to the requested velocity
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      
      if(d_matls.present(gmass, c)) { // only update if shares cell with all requested materials

        for(int n = 0; n < numMatls; n++){ // update rigid body here
          Vector rigid_vel = requested_velocity;
          if(rigid_velocity) {
            rigid_vel = gvelocity[d_material][c];
            if(n==d_material) continue; // compatibility with old mode, where rigid velocity doesnt change material 0
          }
          
          // set each velocity component being modified to a new velocity
          Vector new_vel( gvelocity[n][c] );
          if(d_direction[0]) new_vel.x( rigid_vel.x() );
          if(d_direction[1]) new_vel.y( rigid_vel.y() );
          if(d_direction[2]) new_vel.z( rigid_vel.z() );
          
          // this is the updated velocity
          gvelocity[n][c] = new_vel;
        }
      }
    }
  }
}

// apply boundary conditions to the interpolated velocity v^k+1
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
    StaticArray<NCVariable<Vector> >      gvelocity_star(numMatls);
    StaticArray<constNCVariable<Vector> > gvelocity(numMatls);
    StaticArray<NCVariable<Vector> >      gacceleration(numMatls);
    StaticArray<NCVariable<double> >      frictionWork(numMatls);

    for(int m=0;m<matls->size();m++){
     int dwi = matls->get(m);
     new_dw->get(gmass[m], lb->gMassLabel,dwi ,patch, Ghost::None, 0);
     new_dw->get(gvelocity[m],lb->gVelocityInterpLabel, dwi,patch, Ghost::None,0); // -> v^k
     new_dw->getModifiable(gvelocity_star[m],  lb->gVelocityStarLabel,   dwi,patch); // -> v*^k+1
     new_dw->getModifiable(gacceleration[m],   lb->gAccelerationLabel,   dwi,patch); // -> a*^k+1/2
     new_dw->getModifiable(frictionWork[m],lb->frictionalWorkLabel,dwi,
                           patch);
    }
    
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    
    // set velocity to appropriate vel
    const double tcurr = d_sharedState->getElapsedTime(); // FIXME: + dt ?
    
    bool  rigid_velocity = true;
    Vector requested_velocity;
    if(tcurr>d_stop_time) {
      rigid_velocity = false;
      requested_velocity = d_vel_after_stop;
    } else if(d_vel_profile.size()>0) {
      rigid_velocity = false;
      requested_velocity = findVelFromProfile(tcurr);
    }
    
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      
      if(d_matls.present(gmass, c)) { // only update if shares cell with all requested materials
        
        for(int  n = 0; n < numMatls; n++){ // also updates material d_material to new velocity.
          Vector rigid_vel = requested_velocity;
          if(rigid_velocity) {
            rigid_vel = gvelocity_star[d_material][c];
            if(n==d_material) continue; // compatibility with rigid motion, doesnt affect matl 0
          }
          
          Vector new_vel( gvelocity_star[n][c] );
          if(n==d_material || d_direction[0]) new_vel.x( rigid_vel.x() );
          if(n==d_material || d_direction[1]) new_vel.y( rigid_vel.y() );
          if(n==d_material || d_direction[2]) new_vel.z( rigid_vel.z() );
          
          gvelocity_star[n][c] =  new_vel;
          gacceleration[n][c]  = (gvelocity_star[n][c]  - gvelocity[n][c])/delT;
        }
      }
    }
  }
}

void SpecifiedBodyContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* ms) 
{
  Task * t = new Task("SpecifiedBodyContact::exMomInterpolated",
                      this, &SpecifiedBodyContact::exMomInterpolated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->modifies(             lb->gVelocityLabel,       mss);
  
  sched->addTask(t, patches, ms);
}

void SpecifiedBodyContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* ms) 
{
  Task * t = new Task("SpecifiedBodyContact::exMomIntegrated", 
                      this, &SpecifiedBodyContact::exMomIntegrated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);    
  t->requires(Task::NewDW, lb->gMassLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityInterpLabel, Ghost::None);
  t->modifies(             lb->gVelocityStarLabel,   mss);
  t->modifies(             lb->gAccelerationLabel,   mss);
  t->modifies(             lb->frictionalWorkLabel,  mss);
  
  sched->addTask(t, patches, ms);
}


