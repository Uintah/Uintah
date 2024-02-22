/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include <Core/Math/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Contact/PenaltyRigidContact.h>
#include <vector>
#include <iostream>
#include <dirent.h>

using namespace Uintah;
using std::vector;
using std::string;

using namespace std;


PenaltyRigidContact::PenaltyRigidContact(const ProcessorGroup* myworld,
                               ProblemSpecP& ps,MaterialManagerP& d_sS,
                               MPMLabel* Mlb, MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_oneOrTwoStep = 1;

  // read a list of values from a file
  ps->get("filename", d_filename);

  ps->require("mu",d_mu);

  ps->getWithDefault("master_material", d_material, 0);
  d_matls.add(d_material); // always need specified material

  ps->getWithDefault("include_rotation", d_includeRotation, false);

//  ps->getWithDefault("ExcludeMaterial", d_excludeMatl, -999);

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
            throw ProblemSetupException(
                       "ERROR: profile file is not monotomically increasing",
                        __FILE__, __LINE__);
          }
          d_vel_profile.push_back(std::pair<double,Vector>(t1,Vector(vx,vy,vz)));
        }
        t0 = t1;
      }
    }
    if(d_vel_profile.size()<2) {
      throw ProblemSetupException(
                            "ERROR: Failed to generate valid velocity profile",
                            __FILE__, __LINE__);
    }
  }

  d_materialManager = d_sS;

  // disable all changes after this time
  ps->getWithDefault("stop_time",d_stop_time,std::numeric_limits<double>::max());
  ps->getWithDefault("velocity_after_stop",d_vel_after_stop, Vector(0,0,0));

  if(flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }
}

PenaltyRigidContact::~PenaltyRigidContact()
{
  // Destructor
}

void PenaltyRigidContact::setContactMaterialAttributes()
{
  MPMMaterial* mpm_matl = 
         (MPMMaterial*) d_materialManager->getMaterial( "MPM",  d_material);
  mpm_matl->setIsRigid(true);
}

void PenaltyRigidContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type", "penalty_rigid");
  contact_ps->appendElement("mu",                  d_mu);
  contact_ps->appendElement("filename",            d_filename);
  contact_ps->appendElement("master_material",     d_material);
  contact_ps->appendElement("stop_time",           d_stop_time);
  contact_ps->appendElement("velocity_after_stop", d_vel_after_stop);
  contact_ps->appendElement("include_rotation",    d_includeRotation);
  contact_ps->appendElement("OneOrTwoStep",        d_oneOrTwoStep);
//  contact_ps->appendElement("ExcludeMaterial",     d_excludeMatl);

  d_matls.outputProblemSpec(contact_ps);

  if(d_filename!="") {
    string udaDir = flag->d_DA->getOutputLocation();

    //  Bulletproofing
    DIR *check = opendir(udaDir.c_str());
    if ( check == nullptr){
      ostringstream warn;
      warn << "ERROR:PenaltyRigidContact The main uda directory does not exist.";
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

void PenaltyRigidContact::exMomInterpolated(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
}

void PenaltyRigidContact::exMomIntegrated(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  simTime_vartype simTime;
  old_dw->get(simTime, lb->simulationTimeLabel);

  Ghost::GhostType  gnone = Ghost::None;

  int numMatls = d_materialManager->getNumMatls( "MPM" );
  ASSERTEQ(numMatls, matls->size());

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  std::vector<constNCVariable<double> > gmass(numMatls);
  std::vector<constNCVariable<double> > gvolume(numMatls);
  std::vector<constNCVariable<Vector> > gtrcontactforce(numMatls);
  std::vector<constNCVariable<Vector> > ginternalForce(numMatls);
  std::vector<constNCVariable<int> >    gInContactMatl(numMatls);
  std::vector<NCVariable<Vector> >      gvelocity_star(numMatls);
  std::vector<bool> PistonMaterial(numMatls);

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

  // If rotation axis is aligned with a ordinal direction,
  // use the exact treatment, otherwise default to the approximate
  int rotation_axis = -99;
  if(d_includeRotation){
    double ROL = requested_omega.length();
    if(fabs(Dot(requested_omega/ROL,Vector(1.,0.,0.))) > 0.99){
      rotation_axis=0;
    } else if(fabs(Dot(requested_omega/ROL,Vector(0.,1.,0.))) > 0.99){
      rotation_axis=1;
    } else if(fabs(Dot(requested_omega/ROL,Vector(0.,0.,1.))) > 0.99){
      rotation_axis=2;
    }
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

//    bool useMinNormalDv = true;
    
//    if(flag->d_doingDissolution && flag->d_currentPhase=="hold"){
//       useMinNormalDv = false;
//    }

    // Retrieve necessary data from DataWarehouse
    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      PistonMaterial[m] = mpm_matl->getIsPistonMaterial();
      int dwi = matls->get(m);
      new_dw->get(gmass[m],          lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],        lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gtrcontactforce[m],lb->gLSContactForceLabel,
                                                         dwi, patch, gnone, 0);
      new_dw->get(gInContactMatl[m], lb->gInContactMatlLabel,
                                                         dwi, patch, gnone, 0);
      new_dw->get(ginternalForce[m], lb->gInternalForceLabel,
                                                         dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                                                         dwi, patch);
    }
    Vector dx = patch->dCell();

    if(flag->d_axisymmetric){
      ostringstream warn;
      warn << "Penalty contact not implemented for axisymmetry\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    Vector reaction_force(0.0,0.0,0.0);
    Vector reaction_torque(0.0,0.0,0.0);

    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      Vector centerOfMassVelocity(0.,0.,0.);
      double centerOfMassMass=0.0; 
      for(int  n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        if(gtrcontactforce[n][c].length2() > 0.0){
          centerOfMassVelocity+=gvelocity_star[n][c] * gmass[n][c];
          centerOfMassMass+= gmass[n][c]; 
        }
      }

      centerOfMassVelocity/=centerOfMassMass;

      Vector rotation_part(0.0,0.0,0.0);
      Vector r(0.0,0.0,0.0);

      if(d_includeRotation){
       Point NodePos = patch->getNodePosition(c);
       Point NewNodePos = NodePos;
       // vector from node to a point on the axis of rotation
       r = NodePos - requested_origin.asPoint();
       if(rotation_axis==0){  //rotation about x-axis
         double posz = NodePos.z() - requested_origin.z();
         double posy = NodePos.y() - requested_origin.y();
         double theta = atan2(posz,posy);
         double thetaPlus = theta+requested_omega[0]*delT;
         double R = sqrt(posy*posy + posz*posz);
         NewNodePos = Point(NodePos.x(),
                            R*cos(thetaPlus)+requested_origin.y(),
                            R*sin(thetaPlus)+requested_origin.z());
       } else if(rotation_axis==1){  //rotation about y-axis
         double posx = NodePos.x() - requested_origin.x();
         double posz = NodePos.z() - requested_origin.z();
         double theta = atan2(posx,posz);
         double thetaPlus = theta+requested_omega[1]*delT;
         double R = sqrt(posz*posz + posx*posx);
         NewNodePos = Point(R*sin(thetaPlus)+requested_origin.x(),
                            NodePos.y(),
                            R*cos(thetaPlus)+requested_origin.z());
       } else if(rotation_axis==2){  //rotation about z-axis
         double posx = NodePos.x() - requested_origin.x();
         double posy = NodePos.y() - requested_origin.y();
         double theta = atan2(posy,posx);
         double thetaPlus = theta+requested_omega[2]*delT;
         double R = sqrt(posx*posx + posy*posy);
         NewNodePos = Point(R*cos(thetaPlus)+requested_origin.x(),
                            R*sin(thetaPlus)+requested_origin.y(),
                            NodePos.z());
       } 
       rotation_part = (NewNodePos - NodePos)/delT;
       if(rotation_axis==-99){
         // normal vector from the axis of rotation to the node
         //Vector axis_norm=requested_omega/(requested_omega.length()+1.e-100);
         //Vector rad = r - Dot(r,axis_norm)*axis_norm;
         rotation_part = Cross(requested_omega,r);
       }
      }

      Vector rigid_vel = rotation_part + requested_velocity;
      if(rigid_velocity) {
        rigid_vel = gvelocity_star[d_material][c];
      }

      // Loop over materials.  Only proceed if velocity field mass
      // is nonzero (not numerical noise) and the difference from
      // the centerOfMassVelocity is nonzero (More than one velocity
      // field is contributing to grid vertex).
      for(int n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        double mass=gmass[n][c];
        if(n==d_material){
          gvelocity_star[n][c]     =rigid_vel;
        }
        if(gtrcontactforce[n][c].length2() > 0.0 &&
           !compare(mass,0.0) /*&& mass!=centerOfMassMass*/){
          // Dv is the change in velocity due to penalty forces at tracers
          Vector Dv = (gtrcontactforce[n][c]/mass)*delT;
          double normalDv = Dv.length();
          // This is an inward facing normal
          Vector normal = Dv/(normalDv+1.e-100);

          // deltaVelocity is the difference in velocity of this material
          // relative to the centerOfMassVelocity
          Vector deltaVelocity=gvelocity_star[n][c] - rigid_vel;
          // Get the normal component of deltaVelocity
          double normalDeltaVel = Dot(deltaVelocity,normal);

          // Take the minimum of these two measures of normal velocity.
          // This prevents the penalty force (normalDv) from pushing too hard,
          // and thereby creating space between the materials.  The
          // force based on the velocity (normalDeltaVel) is just enough 
          // to bring the material's velocity to the center of mass velocity.
//          if(!compare(mass-centerOfMassMass,0.0)){
//            normalDv = Min(fabs(normalDv), fabs(normalDeltaVel));
//          } else {
//            normalDv = fabs(normalDv);
//          }
//          if(useMinNormalDv){
            normalDv = Min(fabs(normalDv), fabs(normalDeltaVel));
//          } else {
//            normalDv = fabs(normalDv);
//          }

          // Change in velocity in the normal direction
          Dv = normalDv*normal;

          // Create a vector that is the normal part of deltaVelocity
          Vector normalXnormaldV = normal*normalDeltaVel;
          // Subtract the normal part from the entire deltaVelocity
          Vector dV_normalDV = deltaVelocity - normalXnormaldV;
          // Compute tangent unit vector in the direction of the
          // non-normal part of deltaVelocity
          Vector surfaceTangent = dV_normalDV/(dV_normalDV.length()+1.e-100);

          double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
          double frictionCoefficient=
                    Min(d_mu,tangentDeltaVelocity/(normalDv + 1.e-100));

          if(PistonMaterial[n] || PistonMaterial[gInContactMatl[n][c]]){
           frictionCoefficient=0.0;
          }

          // Change in velocity in the tangential direction
          Dv -= surfaceTangent*frictionCoefficient*normalDv;

          // Define contact algorithm imposed strain, find maximum
          Vector epsilon=(Dv/dx)*delT;
          double epsilon_max=
            Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
          if(!compare(epsilon_max,0.0)){
            // Scale velocity change if contact imposed strain is too large
            double ff=Min(epsilon_max,.1)/epsilon_max;
            Dv=Dv*ff;
          }
          if(n!=d_material){
            gvelocity_star[n][c]    +=Dv;
            reaction_force  -= ginternalForce[n][c];
            reaction_torque += Cross(r,gmass[n][c]*(Dv)/delT);
          }
        }   // if gtrcontactforce>0
      }     // matls
    }       // nodeiterator
    new_dw->put(sumvec_vartype(reaction_force),  lb->RigidReactionForceLabel);
    new_dw->put(sumvec_vartype(reaction_torque), lb->RigidReactionTorqueLabel);
  } // patches
}

void PenaltyRigidContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                        const PatchSet* patches,
                                                        const MaterialSet* ms)
{
  Task * t = scinew Task("Penalty::exMomInterpolated", 
                      this, &PenaltyRigidContact::exMomInterpolated);
  sched->addTask(t, patches, ms);
}

void PenaltyRigidContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms) 
{
  Task * t = scinew Task("Penalty::exMomIntegrated", 
                      this, &PenaltyRigidContact::exMomIntegrated);

  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gMassLabel,                  Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,                Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gLSContactForceLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gInContactMatlLabel,         Ghost::None);

  t->modifies(             lb->gVelocityStarLabel,  mss);
  t->computes(lb->RigidReactionForceLabel);
  t->computes(lb->RigidReactionTorqueLabel);

  sched->addTask(t, patches, ms);
}
