/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// SpecifiedBodyContact.cc
#include <CCA/Components/MPM/Contact/SpecifiedBodyContact.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <Core/Containers/StaticArray.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <numeric>
using std::cerr;

using namespace std;
using namespace Uintah;

SpecifiedBodyContact::SpecifiedBodyContact(const ProcessorGroup* myworld,
                                           ProblemSpecP& ps,
                                           SimulationStateP& d_sS, 
                                           MPMLabel* Mlb, MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  // read a list of values from a file
  ps->get("filename", d_filename);
  
  IntVector defaultDir(0,0,1);
  ps->getWithDefault("direction",d_direction, defaultDir);
  
  ps->getWithDefault("master_material", d_material, 0);
  d_matls.add(d_material); // always need specified material

  d_vol_const=0.;
  ps->get("volume_constraint",d_vol_const);

  ps->getWithDefault("normal_only", d_NormalOnly, false);

  if(d_filename!="") {
    std::ifstream is(d_filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR: opening MPM rigid motion file '"+d_filename+"'\nFailed to find profile file",
                                  __FILE__, __LINE__);
    }
    double t0(-1.e9);
    while(is) {
        double t1;
        double vx, vy, vz;
        is >> t1 >> vx >> vy >> vz;
        if(is) {
            if(t1<=t0){
              throw ProblemSetupException("ERROR: profile file is not monotomically increasing", __FILE__, __LINE__);
            }
            d_vel_profile.push_back( std::pair<double,Vector>(t1,Vector(vx,vy,vz)) );
        }
        t0 = t1;
    }
    if(d_vel_profile.size()<2) {
        throw ProblemSetupException("ERROR: Failed to generate valid velocity profile", __FILE__, __LINE__);
    }
  }
  
  // disable all changes after this time
  ps->getWithDefault("stop_time",d_stop_time, std::numeric_limits<double>::max());
  ps->getWithDefault("velocity_after_stop",d_vel_after_stop, Vector(0,0,0));
  
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
  if(flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }
}

SpecifiedBodyContact::~SpecifiedBodyContact()
{
}

void SpecifiedBodyContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type","specified");
  contact_ps->appendElement("filename",d_filename);
  contact_ps->appendElement("direction",d_direction);
  contact_ps->appendElement("master_material",d_material);
  contact_ps->appendElement("stop_time",d_stop_time);
  contact_ps->appendElement("velocity_after_stop",d_vel_after_stop);
  contact_ps->appendElement("volume_constraint",d_vol_const);

  d_matls.outputProblemSpec(contact_ps);
}


// find velocity from table of values
Vector
SpecifiedBodyContact::findVelFromProfile(double t) const
{
  int smin = 0;
  int smax = (int)(d_vel_profile.size())-1;
  double tmin = d_vel_profile[0].first;
  double tmax = d_vel_profile[smax].first;
  if(t<=tmin) {
      return d_vel_profile[0].second;
  }
  else if(t>=tmax) {
      return d_vel_profile[smax].second;
  }
  else {
      // bisection search on table
      // could probably speed this up by keeping copy of last successful
      // search, and looking at that point and a couple to the right
      //
      while (smax>smin+1) {
          int smid = (smin+smax)/2;
          if(d_vel_profile[smid].first<t){
            smin = smid;
          }
          else{
            smax = smid;
          }
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
#if 0
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
    Vector requested_velocity( 0.0, 0.0, 0.0 );
    if(tcurr>d_stop_time) {
      requested_velocity = d_vel_after_stop;
      rigid_velocity = false;
    } else if(d_vel_profile.size()>0) {
      requested_velocity = findVelFromProfile(tcurr);
      rigid_velocity  = false;
    }
    
    // Set each field's velocity equal to the requested velocity
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter; 

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
        if(!compare(gmass[d_material][c],0.)){
          gvelocity[n][c] = new_vel;
        }
      }
    }
  }
#endif
}

// apply boundary conditions to the interpolated velocity v^k+1
void SpecifiedBodyContact::exMomIntegrated(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gan   = Ghost::AroundNodes;

  int numMatls = d_sharedState->getNumMPMMatls();

  // Retrieve necessary data from DataWarehouse
  StaticArray<constNCVariable<double> > gmass(numMatls);
  StaticArray<NCVariable<Vector> >      gvelocity_star(numMatls);
  StaticArray<constNCVariable<Vector> > gvelocity(numMatls);
  StaticArray<constNCVariable<Vector> > ginternalForce(numMatls);
  StaticArray<constNCVariable<double> > gvolume(numMatls);
  NCVariable<Vector>                    gsurfnorm;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight,         lb->NC_CCweightLabel,  0, patch, gnone, 0);

    for(int m=0;m<matls->size();m++){
     int dwi = matls->get(m);
     new_dw->get(gmass[m],          lb->gMassLabel,         dwi,patch,gnone,0);
     new_dw->get(ginternalForce[m], lb->gInternalForceLabel,dwi,patch,gnone,0);
     new_dw->get(gvolume[m],        lb->gVolumeLabel,       dwi,patch,gnone,0);
     new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,dwi,patch);
    }

    // Compute the normals for the rigid material
   if(d_NormalOnly){
     new_dw->allocateAndPut(gsurfnorm,lb->gSurfNormLabel,d_material,patch);
     gsurfnorm.initialize(Vector(0.0,0.0,0.0));

     ParticleSubset* pset = old_dw->getParticleSubset(d_material, patch,
                                                      gan, NGP, lb->pXLabel);
     constParticleVariable<Point> px;
     constParticleVariable<double> pmass, pvolume;
     constParticleVariable<Matrix3> psize;
     constParticleVariable<Matrix3> deformationGradient;

     old_dw->get(px,                  lb->pXLabel,                  pset);
     old_dw->get(pmass,               lb->pMassLabel,               pset);
     old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
     old_dw->get(psize,               lb->pSizeLabel,               pset);
     old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

     ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
     vector<IntVector> ni(interpolator->size());
     vector<double> S(interpolator->size());
     vector<Vector> d_S(interpolator->size());
     string interp_type = flag->d_interpolator_type;
     double oodx[3];
     oodx[0] = 1.0/dx.x();
     oodx[1] = 1.0/dx.y();
     oodx[2] = 1.0/dx.z();

     if(flag->d_axisymmetric){
      for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
        particleIndex idx = *it;

        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                  deformationGradient[idx]);
        double rho = pmass[idx]/pvolume[idx];

         for(int k = 0; k < flag->d_8or27; k++) {
           if (patch->containsNode(ni[k])){
             Vector G(d_S[k].x(),d_S[k].y(),0.0);
             gsurfnorm[ni[k]] += rho * G;
           } // if
         }   // for
      }      // for
     } else {
      for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
        particleIndex idx = *it;

        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                  deformationGradient[idx]);

         for(int k = 0; k < flag->d_8or27; k++) {
           if (patch->containsNode(ni[k])){
             Vector grad(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                         d_S[k].z()*oodx[2]);
             gsurfnorm[ni[k]] += pmass[idx] * grad;
           }  // if
         }    // for
      }       // for
     }          // else

     MPMBoundCond bc;
     bc.setBoundaryCondition(patch,d_material,"Symmetric",
                             gsurfnorm,interp_type);

     for(NodeIterator iter=patch->getExtraNodeIterator();
                     !iter.done();iter++){
       IntVector c = *iter;
       double length = gsurfnorm[c].length();
       if(length>1.0e-15){
          gsurfnorm[c] = gsurfnorm[c]/length;
       }
     }
   } // if(d_NormalOnly)

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    
    // set velocity to appropriate vel
    const double tcurr = d_sharedState->getElapsedTime(); // FIXME: + dt ?
    
    bool  rigid_velocity = true;
    Vector requested_velocity(0.0, 0.0, 0.0);
    if(tcurr>d_stop_time) {
      rigid_velocity = false;
      requested_velocity = d_vel_after_stop;
    } else if(d_vel_profile.size()>0) {
      rigid_velocity = false;
      requested_velocity = findVelFromProfile(tcurr);
    }

    Vector reaction_force(0.0,0.0,0.0);

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter; 
      
      // Determine nodal volume
      double totalNodalVol=0.0;
      for(int  n = 0; n < numMatls; n++){
        totalNodalVol+=gvolume[n][c]*8.0*NC_CCweight[c];
      }

      for(int  n = 0; n < numMatls; n++){ // also updates material d_material to new velocity.
        Vector rigid_vel = requested_velocity;
        if(rigid_velocity) {
          rigid_vel = gvelocity_star[d_material][c];
          if(n==d_material){
             continue; // compatibility with rigid motion, doesnt affect matl 0
          }
        }

        Vector new_vel(gvelocity_star[n][c]);
        if(d_NormalOnly){
          Vector normal = gsurfnorm[c];
          double normalDeltaVel = Dot(normal,(gvelocity_star[n][c]-rigid_vel));
          if(normalDeltaVel < 0.0){
            Vector normal_normaldV = normal*normalDeltaVel;
            new_vel = gvelocity_star[n][c] - normal_normaldV;
          }
        }
        else{
          new_vel = gvelocity_star[n][c];
          if(n==d_material || d_direction[0]) new_vel.x( rigid_vel.x() );
          if(n==d_material || d_direction[1]) new_vel.y( rigid_vel.y() );
          if(n==d_material || d_direction[2]) new_vel.z( rigid_vel.z() );
        }

        if (!compare(gmass[d_material][c], 0.)
        && (totalNodalVol/cell_vol) > d_vol_const){
          Vector old_vel = gvelocity_star[n][c];
          gvelocity_star[n][c] =  new_vel;
          //reaction_force += gmass[n][c]*(new_vel-old_vel)/delT;
          reaction_force -= ginternalForce[n][c];
        }  // if
      }    // for matls
    }      // for Node Iterator
    new_dw->put(sumvec_vartype(reaction_force), lb->RigidReactionForceLabel);
  }
}

void SpecifiedBodyContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                             const PatchSet* patches,
                                             const MaterialSet* ms) 
{
#if 0
  Task * t = scinew Task("SpecifiedBodyContact::exMomInterpolated",
                      this, &SpecifiedBodyContact::exMomInterpolated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->modifies(             lb->gVelocityLabel,       mss);
  
  sched->addTask(t, patches, ms);
#endif
}

void SpecifiedBodyContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
                                             const PatchSet* patches,
                                             const MaterialSet* ms) 
{
  Task * t = scinew Task("SpecifiedBodyContact::exMomIntegrated", 
                      this, &SpecifiedBodyContact::exMomIntegrated);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);    
  t->requires(Task::NewDW, lb->gMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel,    Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,Ghost::None);

  Ghost::GhostType  gan   = Ghost::AroundNodes;

  if(d_NormalOnly){
   t->requires(Task::OldDW, lb->pXLabel,           gan, NGP);
   t->requires(Task::OldDW, lb->pMassLabel,        gan, NGP);
   t->requires(Task::OldDW, lb->pVolumeLabel,      gan, NGP);
   t->requires(Task::OldDW, lb->pStressLabel,      gan, NGP);
   t->requires(Task::OldDW, lb->pSizeLabel,        gan, NGP);
   t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gan, NGP);
   t->computes(lb->gSurfNormLabel);
  }

  t->modifies(             lb->gVelocityStarLabel,   mss);
  t->computes(lb->RigidReactionForceLabel);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}
