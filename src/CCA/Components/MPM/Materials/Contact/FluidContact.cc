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


#include <CCA/Components/MPM/Materials/Contact/FluidContact.h>

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
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/HydroMPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <vector>
#include <iostream>

using namespace Uintah;
using std::vector;
using std::string;

using namespace std;


FluidContact::FluidContact(const ProcessorGroup* myworld,
                                 MaterialManagerP& d_sS,
                                 MPMLabel* Mlb, HydroMPMLabel* Hlb, MPMFlags* MFlag)
{
  // Constructor  

  d_sharedState = d_sS;

  d_vol_const=0.;
  d_sepFac=9.9e99;  // Default to large number to provide no constraint
  d_compColinearNorms=true;

  // Can add as input parameter too, but hardcoded for now
  // Rigid material
  d_rigid_material = 0;

  if(flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }
}

FluidContact::~FluidContact()
{
  // Destructor
}

void FluidContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type","fluid");
  d_matls.outputProblemSpec(contact_ps);
}

void FluidContact::exMomInterpolated(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{ 
//   // Setting the fluid velocity equal to the rigid velocity,
//   // to satisfy the no-slip boundary condition of impermeable wall-fluid contact
//   // in computational fluid dynamics

//   Ghost::GhostType  gan   = Ghost::AroundNodes;
//   Ghost::GhostType  gnone = Ghost::None;

//   int numMatls = d_sharedState->getNumMPMMatls();
//   ASSERTEQ(numMatls, matls->size());

//   // Need access to all velocity fields at once
//   std::vector<constNCVariable<double> >  gmass(numMatls);
//   std::vector<constNCVariable<double> >  gvolume(numMatls);
//   std::vector<NCVariable<Point> >        gposition(numMatls);
//   std::vector<NCVariable<Vector> >       gvelocity(numMatls);
//   std::vector<NCVariable<Vector> >       gfluidvelocity(numMatls);
//   std::vector<NCVariable<Vector> >       gsurfnorm(numMatls);
//   std::vector<NCVariable<Matrix3> >      gstress(numMatls);

//   for(int p=0;p<patches->size();p++){
//     const Patch* patch = patches->get(p);
//     Vector dx = patch->dCell();
//     double cell_vol = dx.x()*dx.y()*dx.z();
//     double oodx[3];
//     oodx[0] = 1.0/dx.x();
//     oodx[1] = 1.0/dx.y();
//     oodx[2] = 1.0/dx.z();
//     constNCVariable<double> NC_CCweight;
//     old_dw->get(NC_CCweight,         lb->NC_CCweightLabel,  0, patch, gnone, 0);

//     ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
//     vector<IntVector> ni(interpolator->size());
//     vector<double> S(interpolator->size());
//     vector<Vector> d_S(interpolator->size());
//     string interp_type = flag->d_interpolator_type;

//     delt_vartype delT;
//     old_dw->get(delT, lb->delTLabel, getLevel(patches));

//     double sepDis=d_sepFac*cbrt(cell_vol);
//     for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
//       IntVector c = *iter;
//       Vector centerOfMassMom(0.,0.,0.);
//       Point centerOfMassPos(0.,0.,0.);
//       double centerOfMassMass=0.0; 
//       double totalNodalVol=0.0; 
//       for(int n = 0; n < numMatls; n++){
//         if(!d_matls.requested(n)) continue;
//         centerOfMassMom+=gvelocity[n][c] * gmass[n][c];
//         centerOfMassPos+=gposition[n][c].asVector() * gmass[n][c];
//         centerOfMassMass+= gmass[n][c]; 
//         totalNodalVol+=gvolume[n][c]*8.0*NC_CCweight[c];
//       }
//       centerOfMassPos/=centerOfMassMass;

//       // Apply Coulomb friction contact
//       // For grid points with mass calculate velocity
//       if(!compare(centerOfMassMass,0.0)){
//         Vector centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

//         // Only apply contact if the node is nearly "full".  There are
//         // two options:

//         // 1. This option uses particle counting
// //        if((totalNodalVol/cell_vol)*(64./totalNearParticles) > d_vol_const){
//         if((totalNodalVol/cell_vol) > d_vol_const){
//           double scale_factor=1.0;

//           // 2. This option uses only cell volumes.  The idea is that a cell 
//           //    is full if (totalNodalVol/cell_vol >= 1.0), and the contraint 
//           //    should only be applied when cells are full.  This logic is used 
//           //    when d_vol_const=0. 
//           //    For d_vol_const > 0 the contact forces are ramped up linearly 
//           //    from 0 for (totalNodalVol/cell_vol <= 1.0-d_vol_const)
//           //    to 1.0 for (totalNodalVol/cell_vol = 1).  
//           //    Ramping the contact influence seems to help remove a "switching"
//           //    instability.  A good value seems to be d_vol_const=.05

//           //      double scale_factor=0.0;
//           //      if(d_vol_const > 0.0){
//           //        scale_factor=
//           //          (totalNodalVol/cell_vol-1.+d_vol_const)/d_vol_const;
//           //        scale_factor=Max(0.0,scale_factor);
//           //      }
//           //      else if(totalNodalVol/cell_vol > 1.0){
//           //        scale_factor=1.0;
//           //      }

//           //      if(scale_factor > 0.0){
//           //        scale_factor=Min(1.0,scale_factor);
//           //      }

//           // Loop over velocity fields.  Only proceed if velocity field mass
//           // is nonzero (not numerical noise) and the difference from
//           // the centerOfMassVelocity is nonzero (More than one velocity
//           // field is contributing to grid vertex).
//           for(int n = 0; n < numMatls; n++){
//             if(!d_matls.requested(n)) continue;
//             double mass=gmass[n][c];
//             Vector deltaVelocity=gvelocity[n][c]-centerOfMassVelocity;
//             if(!compare(mass/centerOfMassMass,0.0)
//             && !compare(mass-centerOfMassMass,0.0)){

//               // Apply frictional contact IF the surface is in compression
//               // OR the surface is stress free and approaching.
//               // Otherwise apply free surface conditions (do nothing).
//               Vector normal = gsurfnorm[n][c];
//               Vector sepvec = (centerOfMassMass/(centerOfMassMass - mass))*
//                               (centerOfMassPos - gposition[n][c]);
// //              double sepscal= Dot(sepvec,normal);
//               double sepscal= sepvec.length();
//               if(sepscal < sepDis){
//               double normalDeltaVel=Dot(deltaVelocity,normal);
//               Vector Dv(0.,0.,0.);
//               double Tn = gnormtraction[n][c];
//               if((Tn < -1.e-12) || 
//                  (normalDeltaVel> 0.0)){

//                 // Simplify algorithm in case where approach velocity
//                 // is in direction of surface normal (no slip).
//                 Vector normal_normaldV = normal*normalDeltaVel;
//                 Vector dV_normalDV = deltaVelocity - normal_normaldV;
//                   // Calculate velocity change needed to enforce contact
//                   Dv=-normal_normaldV;

//                 // Define contact algorithm imposed strain, find maximum
//                 Vector epsilon=(Dv/dx)*delT;
//                 double epsilon_max=
//                   Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
//                 if(!compare(epsilon_max,0.0)){
//                   epsilon_max *= Max(1.0, mass/(centerOfMassMass-mass));

//                   // Scale velocity change if contact algorithm
//                   // imposed strain is too large.
//                   double ff=Min(epsilon_max,.5)/epsilon_max;
//                   Dv=Dv*ff;
//                 }
//                 Dv=scale_factor*Dv;
//                 gvelocity[n][c]+=Dv;
//                 // if (flag->d_coupledflow && n!=0) gfluidvelocity[n][c] -= normal_normaldV; 
//               }  // if traction
//              }   // if sepscal
//             }    // if !compare && !compare
//           }      // matls
//         }        // if (volume constraint)
//       }          // if(!compare(centerOfMassMass,0.0))
//     }            // NodeIterator

//     delete interpolator;
//   }  // patches
  
}

void FluidContact::exMomIntegrated(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  // Need to check whether we have null contact


  Ghost::GhostType  gnone = Ghost::None;

  // Setting the fluid velocity equal to the rigid velocity,
  // to satisfy the no-slip boundary condition of impermeable wall-fluid contact
  // in computational fluid dynamics
  int numMatls = d_sharedState->getNumMatls();
  ASSERTEQ(numMatls, matls->size());

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  std::vector<constNCVariable<double> > gmass(numMatls);
  std::vector<constNCVariable<double> > gvolume(numMatls);
  // std::vector<constNCVariable<Point> >  gposition(numMatls);
  std::vector<constNCVariable<Vector> > gvelocity(numMatls);
  std::vector<constNCVariable<Vector> > gfluidvelocity(numMatls);
  std::vector<constNCVariable<Vector> > gsurfnorm(numMatls);
  
  std::vector<NCVariable<Vector> >      gfluidvelocity_star(numMatls);
  std::vector<NCVariable<Vector> >      gfluidacceleration(numMatls);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    //double oodx[3];
    //oodx[0] = 1.0/dx.x();
    //oodx[1] = 1.0/dx.y();
    //oodx[2] = 1.0/dx.z();
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight,         lb->NC_CCweightLabel,  0, patch, gnone, 0);
    // Retrieve necessary data from DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],       lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->get(gsurfnorm[m],   lb->gSurfNormLabel,    dwi, patch, gnone, 0);
      // new_dw->get(gposition[m],   lb->gPositionLabel,    dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],     lb->gVolumeLabel,      dwi, patch, gnone, 0);
      new_dw->get(gvelocity[m],   lb->gVelocityLabel,    dwi, patch, gnone, 0);
      new_dw->get(gfluidvelocity[m],   Hlb->gFluidVelocityLabel,    dwi, patch, gnone, 0);
      if (m != d_rigid_material) {
        new_dw->getModifiable(gfluidvelocity_star[m], Hlb->gFluidVelocityStarLabel,
                              dwi, patch);
        new_dw->getModifiable(gfluidacceleration[m], Hlb->gFluidAccelerationLabel,
                              dwi, patch);
      }
    }

    // The normals are already computed if friction contact
    // It is stored in gsurfnorm

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      Vector centerOfMassMom(0.,0.,0.);
      Point centerOfMassPos(0.,0.,0.);
      double centerOfMassMass=0.0; 
      double totalNodalVol=0.0; 
      for(int n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        centerOfMassMom+=gvelocity[n][c] * gmass[n][c];
        centerOfMassMass+= gmass[n][c]; 
        totalNodalVol+=gvolume[n][c]*8.0*NC_CCweight[c];
      }
      centerOfMassPos/=centerOfMassMass;
      
      for(int n = 0; n < numMatls; n++){
        // The rigid material does not change
        if (n == d_rigid_material) continue;
        
        // The rigid velocity
        Vector rigid_vel = gvelocity[d_rigid_material][c];

        if (!compare(gmass[d_rigid_material][c], 0.) &&
            (totalNodalVol/cell_vol)>d_vol_const){
              gfluidvelocity_star[n][c] = gfluidvelocity[n][c] - Dot(gsurfnorm[n][c], gfluidvelocity[n][c]-rigid_vel) * gsurfnorm[n][c];
              gfluidacceleration[n][c] = (gfluidvelocity_star[n][c] - gfluidvelocity[n][c]) / delT;
            }
      }
    }           // nodeiterator
  }
}

void FluidContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                          const PatchSet* patches,
                                                          const MaterialSet* ms)
{
  Task * t = scinew Task("FluidContact::exMomInterpolated", 
              this, &FluidContact::exMomInterpolated);

  Ghost::GhostType  gp;
  int ngc_p;
  //getParticleGhostLayer(gp, ngc_p);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();

  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::OldDW, lb->pXLabel,           gp, ngc_p);
  t->requires(Task::OldDW, lb->pMassLabel,        gp, ngc_p);
  t->requires(Task::OldDW, lb->pVolumeLabel,      gp, ngc_p);
  t->requires(Task::OldDW, lb->pStressLabel,      gp, ngc_p);
  t->requires(Task::OldDW, lb->pSizeLabel,        gp, ngc_p);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gp, ngc_p);
  t->requires(Task::NewDW, lb->gMassLabel,        Ghost::AroundNodes, 1);
  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,Ghost::None);
  // t->computes(lb->gStressLabel); // Will this influence pore pressure?
  t->modifies(Hlb->gFluidVelocityLabel, mss);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}

void FluidContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* ms) 
{
  Task * t = scinew Task("Fluid::exMomIntegrated", 
                      this, &FluidContact::exMomIntegrated);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,Ghost::None);
  // t->requires(Task::NewDW, lb->gSurfNormLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  // t->requires(Task::NewDW, lb->gPositionLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,         Ghost::None);
  t->requires(Task::NewDW, Hlb->gFluidVelocityLabel,         Ghost::None);
  t->modifies(             Hlb->gFluidVelocityStarLabel,  mss);
  t->modifies(             Hlb->gFluidAccelerationLabel,  mss);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}
