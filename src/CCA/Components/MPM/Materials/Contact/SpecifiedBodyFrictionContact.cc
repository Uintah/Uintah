/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

// SpecifiedBodyFrictionContact.cc
#include <CCA/Components/MPM/Materials/Contact/SpecifiedBodyFrictionContact.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>

#include <vector>
#include <fstream>
#include <limits>
#include <dirent.h>
using std::cerr;

using namespace std;
using namespace Uintah;

SpecifiedBodyFrictionContact::SpecifiedBodyFrictionContact(const ProcessorGroup* myworld,
                                           ProblemSpecP& ps,
                                           MaterialManagerP& d_sS, 
                                           MPMLabel* Mlb, MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  // read a list of values from a file
  ps->get("filename", d_filename);
  ps->require("mu",d_mu);

  ps->getWithDefault("master_material", d_material, 0);
  d_matls.add(d_material); // always need specified material

  ps->getWithDefault("include_rotation", d_includeRotation, false);

  if(d_filename!="") {
    std::ifstream is(d_filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR: opening MPM rigid motion file '"+d_filename+"'\nFailed to find profile file",
                                  __FILE__, __LINE__);
    }

    double t0(-1.e9);
    if(d_includeRotation){
      while(is) {
        double t1;
        double vx, vy, vz, ox, oy, oz, wx, wy, wz;
        is >> t1 >> vx >> vy >> vz >> ox >> oy >> oz >> wx >> wy >> wz;
        if(is) {
         if(t1<=t0){
           throw ProblemSetupException("ERROR: profile file is not monotomically increasing", __FILE__, __LINE__);
         }
         d_vel_profile.push_back(std::pair<double,Vector>(t1,Vector(vx,vy,vz)));
         d_rot_profile.push_back(std::pair<double,Vector>(t1,Vector(wx,wy,wz)));
         d_ori_profile.push_back(std::pair<double,Vector>(t1,Vector(ox,oy,oz)));
        }
        t0 = t1;
      }
    } else {
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
    }
    if(d_vel_profile.size()<2) {
        throw ProblemSetupException("ERROR: Failed to generate valid velocity profile", __FILE__, __LINE__);
    }
  }
  
  // disable all changes after this time
  ps->getWithDefault("stop_time",d_stop_time, std::numeric_limits<double>::max());
  ps->getWithDefault("velocity_after_stop",d_vel_after_stop, Vector(0,0,0));
  
  d_materialManager = d_sS;

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


SpecifiedBodyFrictionContact::~SpecifiedBodyFrictionContact()
{
}

void SpecifiedBodyFrictionContact::setContactMaterialAttributes()
{
  MPMMaterial* mpm_matl = 
         (MPMMaterial*) d_materialManager->getMaterial( "MPM",  d_material);
  mpm_matl->setIsRigid(true);
}

void SpecifiedBodyFrictionContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type","specified_friction");
  contact_ps->appendElement("filename",           d_filename);
  contact_ps->appendElement("master_material",    d_material);
  contact_ps->appendElement("stop_time",          d_stop_time);
  contact_ps->appendElement("mu",                 d_mu);
  contact_ps->appendElement("velocity_after_stop",d_vel_after_stop);
  contact_ps->appendElement("include_rotation",   d_includeRotation);

  d_matls.outputProblemSpec(contact_ps);

  if(d_filename!="") {
    string udaDir = flag->d_DA->getOutputLocation();

    //  Bulletproofing
    DIR *check = opendir(udaDir.c_str());
    if ( check == nullptr){
      ostringstream warn;
      warn << "ERROR:SpecifiedBodyFrictionContact The main uda directory does not exist.";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    closedir(check);

    ostringstream fname;
    fname << udaDir << "/"<<  d_filename;
    string filename = fname.str();

    std::ofstream fp(filename.c_str());

    int smax = (int)(d_vel_profile.size());

    if(d_includeRotation){
      for(int i=0;i<smax;i++){
        fp << d_vel_profile[i].first << " " 
           << d_vel_profile[i].second.x() << " " 
           << d_vel_profile[i].second.y() << " " 
           << d_vel_profile[i].second.z() << " " 
           << d_ori_profile[i].second.x() << " " 
           << d_ori_profile[i].second.y() << " " 
           << d_ori_profile[i].second.z() << " " 
           << d_rot_profile[i].second.x() << " " 
           << d_rot_profile[i].second.y() << " " 
           << d_rot_profile[i].second.z() << endl;
      }
    } else {
      for(int i=0;i<smax;i++){
        fp << d_vel_profile[i].first << " " 
           << d_vel_profile[i].second.x() << " " 
           << d_vel_profile[i].second.y() << " " 
           << d_vel_profile[i].second.z() <<  endl;
      }
    }
  }
}

// apply boundary conditions to interpolated velocity v^k
void SpecifiedBodyFrictionContact::exMomInterpolated(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* matls,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
}

// apply boundary conditions to the interpolated velocity v^k+1
void SpecifiedBodyFrictionContact::exMomIntegrated(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  simTime_vartype simTime;
  old_dw->get(simTime, lb->simulationTimeLabel);

  Ghost::GhostType  gnone = Ghost::None;
  int numMatls = d_materialManager->getNumMatls( "MPM" );

  // Retrieve necessary data from DataWarehouse
  std::vector<constNCVariable<double> > gmass(numMatls);
  std::vector<NCVariable<Vector> >      gvelocity_star(numMatls);
  std::vector<constNCVariable<Vector> > gvelocity(numMatls);
  std::vector<constNCVariable<Vector> > ginternalForce(numMatls);
  std::vector<constNCVariable<double> > gvolume(numMatls);
  std::vector<constNCVariable<double> > gmatlprominence(numMatls);    

  // per-matl 
  vector<Vector> reaction_force(  numMatls, Vector(0.0) );
  vector<Vector> reaction_torque( numMatls, Vector(0.0) );

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    constNCVariable<double> NC_CCweight;
    constNCVariable<int>    alphaMaterial;
    constNCVariable<Vector> normAlphaToBeta;
    old_dw->get(NC_CCweight,       lb->NC_CCweightLabel,     0,patch, gnone, 0);
    new_dw->get(alphaMaterial,     lb->gAlphaMaterialLabel,  0,patch, gnone, 0);
    new_dw->get(normAlphaToBeta,   lb->gNormAlphaToBetaLabel,0,patch, gnone, 0);

    for(int m=0;m<matls->size();m++){
     int dwi = matls->get(m);
     new_dw->get(gmass[m],          lb->gMassLabel,         dwi,patch,gnone, 0);
     new_dw->get(ginternalForce[m], lb->gInternalForceLabel,dwi,patch,gnone, 0);
     new_dw->get(gvolume[m],        lb->gVolumeLabel,       dwi,patch,gnone, 0);
     new_dw->get(gmatlprominence[m],lb->gMatlProminenceLabel,
                                                            dwi,patch,gnone, 0);
     new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,dwi,patch);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    
    // rigid_velocity just means that the master_material's initial velocity
    // remains constant through the simulation, until d_stop_time is reached.
    // If the velocity comes from a profile specified in a file, or after
    // d_stop_time, rigid_velocity is false
    bool  rigid_velocity = true;
    Vector requested_velocity(0.0, 0.0, 0.0);
    Vector requested_omega(0.0, 0.0, 0.0);
    Vector requested_origin(0.0, 0.0, 0.0);
    if(simTime>d_stop_time) {
      rigid_velocity = false;
      requested_velocity = d_vel_after_stop;
    } else if(d_vel_profile.size()>0) {
      rigid_velocity = false;
      requested_velocity = findValFromProfile(simTime, d_vel_profile);
      if(d_includeRotation){
        requested_origin = findValFromProfile(simTime, d_ori_profile);
        requested_omega  = findValFromProfile(simTime, d_rot_profile);
      }
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter; 
      
      int alpha=alphaMaterial[c];
      Point NodePos = patch->getNodePosition(c);
      Vector r = NodePos - requested_origin.asPoint();
      Vector rigid_vel = Cross(requested_omega,r) + requested_velocity;
      if(rigid_velocity) {
        rigid_vel = gvelocity_star[d_material][c];
      }
      if(alpha>=0){  // Only work on nodes where alpha!=-99
        for(int  n = 0; n < numMatls; n++){
          if(!d_matls.requested(n)) continue;
          Vector new_vel = rigid_vel;

          if(n==d_material){
            gvelocity_star[n][c] =  new_vel;
          } else if (!compare(gmass[d_material][c], 0.) &&
                     !compare(gmass[n][c],0)) {
            double separation = gmatlprominence[n][c] -
                                gmatlprominence[alpha][c];
            if(separation <= 0.0){
              Vector old_vel = gvelocity_star[n][c];

              Vector deltaVelocity=gvelocity_star[n][c] - new_vel;
              Vector normal = -1.0*normAlphaToBeta[c];
              double normalDeltaVel=Dot(deltaVelocity,normal);
              Vector Dv(0.,0.,0.);
              if(normalDeltaVel > 0.0){

                Vector normal_normaldV = normal*normalDeltaVel;
                Vector dV_normalDV = deltaVelocity - normal_normaldV;
                Vector surfaceTangent = 
                                   dV_normalDV/(dV_normalDV.length()+1.e-100);
                double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                double frictionCoefficient=
                  Min(d_mu,tangentDeltaVelocity/(fabs(normalDeltaVel)+1.e-100));
                // Calculate velocity change needed to enforce contact
                Dv = -normal_normaldV
                     -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

                gvelocity_star[n][c]    +=Dv;

                //reaction_force += gmass[n][c]*(new_vel-old_vel)/delT;
//              reaction_force  -= ginternalForce[n][c];
//              reaction_torque += Cross(r,gmass[n][c]*(new_vel-old_vel)/delT);
                reaction_force[n] -=ginternalForce[n][c];
                reaction_torque[n]+=Cross(r,gmass[n][c]*(new_vel-old_vel)/delT);
              }  // if normalDeltaVel > 0
            }  // if separation
          }  // if mass of both matls>0
        }    // for matls
      } else{  // alpha!=0
        for(int  n = 0; n < numMatls; n++){
          if(n==d_material && gmass[n][c]>1.e-99){
            gvelocity_star[n][c] =  rigid_vel;
          }
        }
      }
    }      // for Node Iterator
  } // loop over patches

  //__________________________________
  //  reduction Vars
  reaction_force[d_material]=Vector(0.0,0.0,0.0);

  for(int  n = 0; n < numMatls; n++){
    if(n!=d_material){
      reaction_force[d_material]+=reaction_force[n];
      reaction_torque[d_material]+=reaction_torque[n];
    }
  }

  for(int  n = 0; n < numMatls; n++){
    int dwi = matls->get(n);

    if( numMatls > 1 ){  // ignore for single matl problems
      new_dw->put( sumvec_vartype(reaction_force[n]),
                                    lb->RigidReactionForceLabel,  nullptr, dwi);
      new_dw->put( sumvec_vartype(reaction_torque[n]),
                                    lb->RigidReactionTorqueLabel, nullptr, dwi);
    }
  }

  new_dw->put( sumvec_vartype( reaction_force[d_material] ),
                                     lb->RigidReactionForceLabel, nullptr, -1 );
  new_dw->put( sumvec_vartype( reaction_torque[d_material] ),
                                     lb->RigidReactionTorqueLabel,nullptr, -1 );
}

void SpecifiedBodyFrictionContact::addComputesAndRequiresInterpolated(
                                                             SchedulerP & sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* ms) 
{
}

void SpecifiedBodyFrictionContact::addComputesAndRequiresIntegrated(
                                                             SchedulerP & sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* ms) 
{
  Task * t = scinew Task("SpecifiedBodyFrictionContact::exMomIntegrated", 
                      this, &SpecifiedBodyFrictionContact::exMomIntegrated);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->simulationTimeLabel);
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gMassLabel,                    Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,                  Ghost::None);
  t->requires(Task::NewDW, lb->gMatlProminenceLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gAlphaMaterialLabel,           Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,      z_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gNormAlphaToBetaLabel, z_matl, Ghost::None);

  t->modifies(             lb->gVelocityStarLabel,   mss);

  //__________________________________
  //  Create reductionMatlSubSet that includes all mss matls
  //  and the global matlsubset
  const MaterialSubset* global_mss = t->getGlobalMatlSubset();

  MaterialSubset* reduction_mss = scinew MaterialSubset();
  reduction_mss->add( global_mss->get(0) );

  unsigned int numMatls = mss->size();

  if( numMatls > 1 ){  // ignore for single matl problems
    for (unsigned int m = 0; m < numMatls; m++ ) {
      reduction_mss->add( mss->get(m) );
    }
  }

  reduction_mss->addReference();

  t->computes( lb->RigidReactionForceLabel,  reduction_mss );
  t->computes( lb->RigidReactionTorqueLabel, reduction_mss );

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference()){
    delete z_matl; // shouln't happen, but...
  }

  if (reduction_mss && reduction_mss->removeReference()){
    delete reduction_mss;
  } 

}

// find velocity from table of values
Vector
SpecifiedBodyFrictionContact::findValFromProfile(double t, 
                                         vector<pair<double, Vector> > profile) const
{
  int smin = 0;
  int smax = (int)(profile.size())-1;
  double tmin = profile[0].first;
  double tmax = profile[smax].first;
  if(t<=tmin) {
      return profile[0].second;
  }
  else if(t>=tmax) {
      return profile[smax].second;
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
      double l  = (profile[smin+1].first-profile[smin].first);
      double xi = (t-profile[smin].first)/l;
      double vx = xi*profile[smin+1].second[0]+(1-xi)*profile[smin].second[0];
      double vy = xi*profile[smin+1].second[1]+(1-xi)*profile[smin].second[1];
      double vz = xi*profile[smin+1].second[2]+(1-xi)*profile[smin].second[2];
      return Vector(vx,vy,vz);
    }
}
