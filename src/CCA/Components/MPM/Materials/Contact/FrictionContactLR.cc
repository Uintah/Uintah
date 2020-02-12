/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Contact/FrictionContactLR.h>
#include <vector>
#include <iostream>

using namespace Uintah;
using std::vector;
using std::string;

using namespace std;


FrictionContactLR::FrictionContactLR(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps,MaterialManagerP& d_sS,
                                 MPMLabel* Mlb,MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_vol_const=0.;
  d_oneOrTwoStep = 2;

  ps->require("mu",d_mu);
  ps->get("volume_constraint",d_vol_const);
  ps->get("OneOrTwoStep",     d_oneOrTwoStep);

  d_materialManager = d_sS;

  if(flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }
}

FrictionContactLR::~FrictionContactLR()
{
  // Destructor
}

void FrictionContactLR::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type", "friction_LR");
  contact_ps->appendElement("mu",                d_mu);
  contact_ps->appendElement("volume_constraint", d_vol_const);
  contact_ps->appendElement("OneOrTwoStep",      d_oneOrTwoStep);
  d_matls.outputProblemSpec(contact_ps);
}

void FrictionContactLR::exMomInterpolated(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  if(d_oneOrTwoStep==2){

   int numMatls = d_materialManager->getNumMatls( "MPM" );
   ASSERTEQ(numMatls, matls->size());

   // Need access to all velocity fields at once
   std::vector<constNCVariable<double> >  gmass(numMatls);
   std::vector<constNCVariable<double> >  gvolume(numMatls);
   std::vector<constNCVariable<double> >  gmatlprominence(numMatls);
   std::vector<NCVariable<Vector> >       gvelocity(numMatls);

   Ghost::GhostType  gnone = Ghost::None;

   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    constNCVariable<double> NC_CCweight;
    constNCVariable<int> alphaMaterial;
    constNCVariable<Vector> normAlphaToBeta;
    old_dw->get(NC_CCweight,      lb->NC_CCweightLabel,     0,patch, gnone, 0);
    new_dw->get(alphaMaterial,    lb->gAlphaMaterialLabel,  0,patch, gnone, 0);
    new_dw->get(normAlphaToBeta,  lb->gNormAlphaToBetaLabel,0,patch, gnone, 0);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // First, calculate the gradient of the mass everywhere
    // normalize it, and stick it in surfNorm
    for(int m=0;m<numMatls;m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],          lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],        lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gmatlprominence[m],lb->gMatlProminenceLabel,
                                                         dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity[m],   lb->gVelocityLabel,      dwi,patch);
    }  // loop over matls

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      Vector centerOfMassVelocity(0.,0.,0.);
      double centerOfMassMass=0.0; 
      double totalNodalVol=0.0; 
      int alpha=alphaMaterial[c];
      // Need to think whether centerOfMass(Stuff) should
      // only include current material and alpha material
      // Why include materials that may be putting mass on the node
      // but aren't near enough to be in proper contact.
      for(int n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        centerOfMassVelocity+=gvelocity[n][c] * gmass[n][c];
        centerOfMassMass+= gmass[n][c]; 
        totalNodalVol+=gvolume[n][c]*8.0*NC_CCweight[c];
      }

      if(alpha>=0){  // Only work on nodes where alpha!=-99
        centerOfMassVelocity/=centerOfMassMass;

        if(flag->d_axisymmetric){
          // Nodal volume isn't constant for axisymmetry
          // volume = r*dr*dtheta*dy  (dtheta = 1 radian)
          double r = min((patch->getNodePosition(c)).x(),.5*dx.x());
          cell_vol =  r*dx.x()*dx.y();
        }

        // Only apply contact if the node is full relative to a constraint
        if((totalNodalVol/cell_vol) > d_vol_const){

          // Loop over materials.  Only proceed if velocity field mass
          // is nonzero (not numerical noise) and the difference from
          // the centerOfMassVelocity is nonzero (More than one velocity
          // field is contributing to grid vertex).
          for(int n = 0; n < numMatls; n++){
           if(!d_matls.requested(n)) continue;
           if(n==alpha) continue;
            double mass=gmass[n][c];
            if(mass>1.e-16){ // There is mass of material beta at this node
              // Check relative separation of the material prominence
              double separation = gmatlprominence[n][c] - 
                                  gmatlprominence[alpha][c];
              // If that separation is negative, the matls have overlapped
//              if(separation <= 0.0){
              if(separation <= 0.01*dx.x()){
               Vector deltaVelocity=gvelocity[n][c] - centerOfMassVelocity;
               Vector normal = -1.0*normAlphaToBeta[c];
               double normalDeltaVel=Dot(deltaVelocity,normal);
               Vector Dv(0.,0.,0.);
               if(normalDeltaVel > 0.0){
                 Vector normal_normaldV = normal*normalDeltaVel;
                 Vector dV_normalDV = deltaVelocity - normal_normaldV;
                 Vector surfaceTangent = dV_normalDV/(dV_normalDV.length()+1.e-100);
                 double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                 double frictionCoefficient=
                         Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel));

                 // Calculate velocity change needed to enforce contact
                 Dv = -normal_normaldV
                   -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

#if 0
                // Define contact algorithm imposed strain, find maximum
                Vector epsilon=(Dv/dx)*delT;
                double epsilon_max=
                  Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
                if(!compare(epsilon_max,0.0)){
                   epsilon_max *= Max(1.0, mass/(centerOfMassMass-mass));
 
                   // Scale velocity change if contact algorithm
                   // imposed strain is too large.
                   double ff=Min(epsilon_max,.5)/epsilon_max;
                   Dv=Dv*ff;
                }
#endif 
                double ff = max(1.0,(.01*dx.x() - separation)/.01*dx.x());
                Dv=Dv*ff;
                Vector DvAlpha = -Dv*gmass[n][c]/gmass[alpha][c];
                gvelocity[n][c]    +=Dv;
                gvelocity[alpha][c]+=DvAlpha;
              } // if (relative velocity) * normal < 0
             }  // if separation
            }   // if !compare && !compare
          }     // matls
        }       // if (volume constraint)
      }         // if(alpha > 0)
    }           // NodeIterator
  }             // patches
 }              // if d_oneOrTwoStep
}

void FrictionContactLR::exMomIntegrated(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;

  int numMatls = d_materialManager->getNumMatls( "MPM" );
  ASSERTEQ(numMatls, matls->size());

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  std::vector<constNCVariable<double> > gmass(numMatls);
  std::vector<constNCVariable<double> > gvolume(numMatls);
  std::vector<constNCVariable<double> > gmatlprominence(numMatls);    
  std::vector<NCVariable<Vector> >      gvelocity_star(numMatls);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    constNCVariable<double> NC_CCweight;
    constNCVariable<int> alphaMaterial;
    constNCVariable<Vector> normAlphaToBeta;
    old_dw->get(NC_CCweight,      lb->NC_CCweightLabel,     0,patch, gnone, 0);
    new_dw->get(alphaMaterial,    lb->gAlphaMaterialLabel,  0,patch, gnone, 0);
    new_dw->get(normAlphaToBeta,  lb->gNormAlphaToBetaLabel,0,patch, gnone, 0);

    // Retrieve necessary data from DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],       lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],     lb->gVolumeLabel,      dwi, patch, gnone, 0);
      new_dw->get(gmatlprominence[m],lb->gMatlProminenceLabel,
                                                         dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                            dwi, patch);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      Vector centerOfMassVelocity(0.,0.,0.);
      double centerOfMassMass=0.0; 
      double totalNodalVol=0.0; 
      int alpha=alphaMaterial[c];
      for(int  n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        centerOfMassVelocity+=gvelocity_star[n][c] * gmass[n][c];
        centerOfMassMass+= gmass[n][c]; 
        totalNodalVol+=gvolume[n][c]*8.0*NC_CCweight[c];
      }

      if(alpha>=0){  // Only work on nodes where alpha!=-99
        centerOfMassVelocity/=centerOfMassMass;
        if(flag->d_axisymmetric){
          // Nodal volume isn't constant for axisymmetry
          // volume = r*dr*dtheta*dy  (dtheta = 1 radian)
          double r = min((patch->getNodePosition(c)).x(),.5*dx.x());
          cell_vol =  r*dx.x()*dx.y();
        }

        // Only apply contact if the node is full relative to a constraint
        if((totalNodalVol/cell_vol) > d_vol_const){

          // Loop over materials.  Only proceed if velocity field mass
          // is nonzero (not numerical noise) and the difference from
          // the centerOfMassVelocity is nonzero (More than one velocity
          // field is contributing to grid vertex).
          for(int n = 0; n < numMatls; n++){
           if(!d_matls.requested(n)) continue;
           if(n==alpha) continue;
            double mass=gmass[n][c];
            if(mass>1.e-16){
              double separation = gmatlprominence[n][c] - 
                                  gmatlprominence[alpha][c];
//              if(separation <= 0.0){
              if(separation <= 0.01*dx.x()){
               Vector deltaVelocity=gvelocity_star[n][c] - centerOfMassVelocity;
               Vector normal = -1.0*normAlphaToBeta[c];
               double normalDeltaVel=Dot(deltaVelocity,normal);
               Vector Dv(0.,0.,0.);
               if(normalDeltaVel > 0.0){

                Vector normal_normaldV = normal*normalDeltaVel;
                Vector dV_normalDV = deltaVelocity - normal_normaldV;
                Vector surfaceTangent = dV_normalDV/(dV_normalDV.length()+1.e-100);
                double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                double frictionCoefficient=
                        Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel));
                // Calculate velocity change needed to enforce contact
                Dv = -normal_normaldV
                     -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

#if 0
                // Define contact algorithm imposed strain, find maximum
                Vector epsilon=(Dv/dx)*delT;
                double epsilon_max=
                  Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
                if(!compare(epsilon_max,0.0)){
                  epsilon_max *= Max(1.0, mass/(centerOfMassMass-mass));

                  // Scale velocity change if contact algorithm
                  // imposed strain is too large.
                  double ff=Min(epsilon_max,.5)/epsilon_max;
                  Dv=Dv*ff;
                }
#endif 
                double ff = max(1.0,(.01*dx.x() - separation)/.01*dx.x());
                Dv=Dv*ff;
                gvelocity_star[n][c]    +=Dv;
                Vector DvAlpha = -Dv*gmass[n][c]/gmass[alpha][c];
                gvelocity_star[alpha][c]+=DvAlpha;
              }  // if (relative velocity) * normal < 0
             }   // if separation
            }    // if mass[beta] > 0
          }      // matls
        }        // if (volume constraint)
      }          // if(alpha > 0)
    }           // nodeiterator
  } // patches
}

void FrictionContactLR::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                        const PatchSet* patches,
                                                        const MaterialSet* ms)
{
  Task * t = scinew Task("Friction::exMomInterpolated", 
                      this, &FrictionContactLR::exMomInterpolated);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gMassLabel,                  Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,                Ghost::None);
  t->requires(Task::NewDW, lb->gMatlProminenceLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gAlphaMaterialLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gNormAlphaToBetaLabel,z_matl,Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,     Ghost::None);
  t->modifies(lb->gVelocityLabel,      mss);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}

void FrictionContactLR::addComputesAndRequiresIntegrated(SchedulerP & sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* ms) 
{
  Task * t = scinew Task("Friction::exMomIntegrated", 
                      this, &FrictionContactLR::exMomIntegrated);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gMassLabel,                  Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,                Ghost::None);
  t->requires(Task::NewDW, lb->gMatlProminenceLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gAlphaMaterialLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,     Ghost::None);
  t->requires(Task::NewDW, lb->gNormAlphaToBetaLabel,z_matl,Ghost::None);
  t->modifies(             lb->gVelocityStarLabel,  mss);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}
