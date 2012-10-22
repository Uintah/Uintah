/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Containers/StaticArray.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/Contact/FrictionContact.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace Uintah;
using std::vector;
using std::string;

using namespace std;


FrictionContact::FrictionContact(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps,SimulationStateP& d_sS,
                                 MPMLabel* Mlb,MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_vol_const=0.;
  
  ps->require("mu",d_mu);
  ps->get("volume_constraint",d_vol_const);

  d_sharedState = d_sS;

  if(flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(flag->d_8or27==27 || flag->d_8or27==64){
    NGP=2;
    NGN=2;
  }
}

FrictionContact::~FrictionContact()
{
  // Destructor
}

void FrictionContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type","friction");
  contact_ps->appendElement("mu",d_mu);
  contact_ps->appendElement("volume_constraint",d_vol_const);
  d_matls.outputProblemSpec(contact_ps);
}


void FrictionContact::exMomInterpolated(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{ 
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;

  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());

  // Need access to all velocity fields at once
  StaticArray<constNCVariable<double> >  gmass(numMatls);
  StaticArray<constNCVariable<double> >  gvolume(numMatls);
  StaticArray<NCVariable<Vector> >       gvelocity(numMatls);
  StaticArray<NCVariable<Vector> >       gsurfnorm(numMatls);
  StaticArray<NCVariable<double> >       frictionWork(numMatls);
  StaticArray<NCVariable<Matrix3> >      gstress(numMatls);
  StaticArray<NCVariable<double> >       gnormtraction(numMatls);

  constNCVariable<double> gm;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight,         lb->NC_CCweightLabel,  0, patch, gnone, 0);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    string interp_type = flag->d_interpolator_type;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // First, calculate the gradient of the mass everywhere
    // normalize it, and stick it in surfNorm
    for(int m=0;m<numMatls;m++){
      int dwi = matls->get(m);

      new_dw->get(gmass[m],           lb->gMassLabel,  dwi, patch, gan,   1);
      new_dw->get(gvolume[m],         lb->gVolumeLabel,dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity[m],  lb->gVelocityLabel,       dwi,patch);
      new_dw->allocateAndPut(gsurfnorm[m], lb->gSurfNormLabel,       dwi,patch);
      new_dw->getModifiable(frictionWork[m],lb->frictionalWorkLabel, dwi,patch);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      constParticleVariable<Point> px;
      constParticleVariable<double> pmass, pvolume;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;

      old_dw->get(px,                  lb->pXLabel,                  pset);
      old_dw->get(pmass,               lb->pMassLabel,               pset);
      old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
      old_dw->get(psize,               lb->pSizeLabel,               pset);
      old_dw->get(deformationGradient,  lb->pDeformationMeasureLabel, pset);

      gsurfnorm[m].initialize(Vector(0.0,0.0,0.0));

      if(!d_matls.requested(m)) continue;

      // Compute the normals for all of the interior nodes
      if(flag->d_axisymmetric){
        for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
          particleIndex idx = *it;

          interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);
          double rho = pmass[idx]/pvolume[idx];

           for(int k = 0; k < flag->d_8or27; k++) {
             if (patch->containsNode(ni[k])){
               Vector G(d_S[k].x(),d_S[k].y(),d_S[k].z()/px[idx].x());
               gsurfnorm[m][ni[k]] += rho * G;
             }
           }
        }
     } else {
        for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
          particleIndex idx = *it;

          interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

           for(int k = 0; k < flag->d_8or27; k++) {
             if (patch->containsNode(ni[k])){
               Vector grad(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                           d_S[k].z()*oodx[2]);
               gsurfnorm[m][ni[k]] += pmass[idx] * grad;
             }
           }
        }
     }
    }  // loop over matls


    for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
       IntVector c = *iter;
       double max_mag = gsurfnorm[0][c].length();
       int max_mag_matl = 0;
       for(int m=1; m<numMatls; m++){
         double mag = gsurfnorm[m][c].length();
         if(mag > max_mag){
            max_mag = mag;
            max_mag_matl = m;
         }
       }  // loop over matls
       for(int m=0; m<numMatls; m++){
        if(m!=max_mag_matl){
          gsurfnorm[m][c] = -gsurfnorm[max_mag_matl][c];
        }
       }  // loop over matls
    }


    for(int m=0;m<numMatls;m++){
      int dwi = matls->get(m);
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gsurfnorm[m],interp_type);

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
         IntVector c = *iter;
         double length = gsurfnorm[m][c].length();
         if(length>1.0e-15){
            gsurfnorm[m][c] = gsurfnorm[m][c]/length;
         }
      }
    }  // loop over matls

    for(int m=0;m<numMatls;m++){
      int dwi = matls->get(m);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pstress, deformationGradient;

      old_dw->get(px,                   lb->pXLabel,                  pset);
      old_dw->get(psize,                lb->pSizeLabel,               pset);
      old_dw->get(deformationGradient,  lb->pDeformationMeasureLabel, pset);
      old_dw->get(pstress,              lb->pStressLabel,             pset);

      new_dw->allocateAndPut(gnormtraction[m],lb->gNormTractionLabel,dwi,patch);
      new_dw->allocateAndPut(gstress[m],      lb->gStressLabel,      dwi,patch);
      gstress[m].initialize(Matrix3(0.0));

      // Next, interpolate the stress to the grid
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx], ni, S, psize[idx],
                                         deformationGradient[idx]);

        // Add each particles contribution to the local mass & velocity
        // Must use the node indices
        for(int k = 0; k < flag->d_8or27; k++) {
          if (patch->containsNode(ni[k]))
            gstress[m][ni[k]] += pstress[idx] * S[k];
        }
      }

      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++){
        IntVector c = *iter;
        Vector norm = gsurfnorm[m][c];
        gnormtraction[m][c]= Dot((norm*gstress[m][c]),norm);
      }
    }  // loop over matls

#if 1
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      Vector centerOfMassMom(0.,0.,0.);
      double centerOfMassMass=0.0; 
      double totalNodalVol=0.0; 
      for(int n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        centerOfMassMom+=gvelocity[n][c] * gmass[n][c];
        centerOfMassMass+= gmass[n][c]; 
        totalNodalVol+=gvolume[n][c]*8.0*NC_CCweight[c];
      }

      // Apply Coulomb friction contact
      // For grid points with mass calculate velocity
      if(!compare(centerOfMassMass,0.0)){
        Vector centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

        if(flag->d_axisymmetric){
          // Nodal volume isn't constant for axisymmetry
          // volume = r*dr*dtheta*dy  (dtheta = 1 radian)
          double r = min((patch->getNodePosition(c)).x(),.5*dx.x());
          cell_vol =  r*dx.x()*dx.y();
        }

        // Only apply contact if the node is nearly "full".  There are
        // two options:

        // 1. This option uses particle counting
//        if((totalNodalVol/cell_vol)*(64./totalNearParticles) > d_vol_const){
        if((totalNodalVol/cell_vol) > d_vol_const){
          double scale_factor=1.0;

          // 2. This option uses only cell volumes.  The idea is that a cell 
          //    is full if (totalNodalVol/cell_vol >= 1.0), and the contraint 
          //    should only be applied when cells are full.  This logic is used 
          //    when d_vol_const=0. 
          //    For d_vol_const > 0 the contact forces are ramped up linearly 
          //    from 0 for (totalNodalVol/cell_vol <= 1.0-d_vol_const)
          //    to 1.0 for (totalNodalVol/cell_vol = 1).  
          //    Ramping the contact influence seems to help remove a "switching"
          //    instability.  A good value seems to be d_vol_const=.05

          //      double scale_factor=0.0;
          //      if(d_vol_const > 0.0){
          //        scale_factor=
          //          (totalNodalVol/cell_vol-1.+d_vol_const)/d_vol_const;
          //        scale_factor=Max(0.0,scale_factor);
          //      }
          //      else if(totalNodalVol/cell_vol > 1.0){
          //        scale_factor=1.0;
          //      }

          //      if(scale_factor > 0.0){
          //        scale_factor=Min(1.0,scale_factor);
          //      }

          // Loop over velocity fields.  Only proceed if velocity field mass
          // is nonzero (not numerical noise) and the difference from
          // the centerOfMassVelocity is nonzero (More than one velocity
          // field is contributing to grid vertex).
          for(int n = 0; n < numMatls; n++){
            if(!d_matls.requested(n)) continue;
            double mass=gmass[n][c];
            Vector deltaVelocity=gvelocity[n][c]-centerOfMassVelocity;
            if(!compare(mass/centerOfMassMass,0.0)
            && !compare(mass-centerOfMassMass,0.0)){

              // Apply frictional contact IF the surface is in compression
              // OR the surface is stress free and approaching.
              // Otherwise apply free surface conditions (do nothing).
              Vector normal = gsurfnorm[n][c];
              double normalDeltaVel=Dot(deltaVelocity,normal);
              Vector Dv(0.,0.,0.);
              double Tn = gnormtraction[n][c];
              if((Tn <  0.0) || 
                 (compare(fabs(Tn),0.0) && normalDeltaVel> 0.0)){

                // Simplify algorithm in case where approach velocity
                // is in direction of surface normal (no slip).
                Vector normal_normaldV = normal*normalDeltaVel;
                Vector dV_normalDV = deltaVelocity - normal_normaldV;
                if(compare(dV_normalDV.length2(),0.0)){

                  // Calculate velocity change needed to enforce contact
                  Dv=-normal_normaldV;
                }

                // General algorithm, including frictional slip.  The
                // contact velocity change and frictional work are both
                // zero if normalDeltaVel is zero.
                else if(!compare(fabs(normalDeltaVel),0.0)){
                  Vector surfaceTangent = dV_normalDV/dV_normalDV.length();
                  double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                  double frictionCoefficient=
                    Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel));

                  // Calculate velocity change needed to enforce contact
                  Dv = -normal_normaldV
                    -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

                  // Calculate work done by the frictional force (only) if
                  // contact slips.  Because the frictional force opposes motion
                  // it is dissipative and should always be negative per the
                  // conventional definition.  However, here it is calculated
                  // as positive (Work=-force*distance).
                  if(compare(frictionCoefficient,d_mu)){
                    if (flag->d_fracture)
                      frictionWork[n][c] += mass*frictionCoefficient
                        * (normalDeltaVel*normalDeltaVel) *
                        (tangentDeltaVelocity/fabs(normalDeltaVel)-
                         frictionCoefficient);
                    else
                      frictionWork[n][c] = mass*frictionCoefficient
                        * (normalDeltaVel*normalDeltaVel) *
                        (tangentDeltaVelocity/fabs(normalDeltaVel)-
                         frictionCoefficient);
                  }
                }

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
                Dv=scale_factor*Dv;
                gvelocity[n][c]+=Dv;
              }  // if traction
            }    // if !compare && !compare
          }      // matls
        }       // if (volume constraint)
      }        // if(!compare(centerOfMassMass,0.0))
    }          // NodeIterator
#endif

    delete interpolator;
  }  // patches
  
}

void FrictionContact::exMomIntegrated(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;

  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  StaticArray<constNCVariable<double> > gmass(numMatls);
  StaticArray<constNCVariable<double> > gvolume(numMatls);
  StaticArray<NCVariable<Vector> >      gvelocity_star(numMatls);
  StaticArray<constNCVariable<double> > normtraction(numMatls);
  StaticArray<NCVariable<double> >      frictionWork(numMatls);
  StaticArray<constNCVariable<Vector> > gsurfnorm(numMatls);    

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight,         lb->NC_CCweightLabel,  0, patch, gnone, 0);

    // Retrieve necessary data from DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],       lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->get(normtraction[m],lb->gNormTractionLabel,dwi, patch, gnone, 0);
      new_dw->get(gsurfnorm[m],   lb->gSurfNormLabel,    dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],     lb->gVolumeLabel,      dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                            dwi, patch);
      new_dw->getModifiable(frictionWork[m], lb->frictionalWorkLabel,
                            dwi, patch);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    double epsilon_max_max=0.0;

    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      Vector centerOfMassMom(0.,0.,0.);
      double centerOfMassMass=0.0; 
      double totalNodalVol=0.0; 
      for(int  n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        double mass = gmass[n][c];
        centerOfMassMom+=gvelocity_star[n][c] * mass;
        centerOfMassMass+= mass; 
        totalNodalVol+=gvolume[n][c]*8.0*NC_CCweight[c];
      }

      // Apply Coulomb friction contact
      // For grid points with mass calculate velocity
      if(!compare(centerOfMassMass,0.0)){
        Vector centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

        if(flag->d_axisymmetric){
          // Nodal volume isn't constant for axisymmetry
          // volume = r*dr*dtheta*dy  (dtheta = 1 radian)
          double r = min((patch->getNodePosition(c)).x(),.5*dx.x());
          cell_vol =  r*dx.x()*dx.y();
        }

        // Only apply contact if the node is nearly "full".  There are
        // two options:

        if((totalNodalVol/cell_vol) > d_vol_const){
          double scale_factor=1.0;

          // 2. This option uses only cell volumes.  The idea is that a cell 
          //    is full if (totalNodalVol/cell_vol >= 1.0), and the contraint 
          //    should only be applied when cells are full.  This logic is used 
          //    when d_vol_const=0. 
          //    For d_vol_const > 0 the contact forces are ramped up linearly 
          //    from 0 for (totalNodalVol/cell_vol <= 1.0-d_vol_const)
          //    to 1.0 for (totalNodalVol/cell_vol = 1).  
          //    Ramping the contact influence seems to help remove a "switching"
          //    instability.  A good value seems to be d_vol_const=.05

          //      double scale_factor=0.0;
          //      if(d_vol_const > 0.0){
          //        scale_factor=
          //          (totalNodalVol/cell_vol-1.+d_vol_const)/d_vol_const;
          //        scale_factor=Max(0.0,scale_factor);
          //      }
          //      else if(totalNodalVol/cell_vol > 1.0){
          //        scale_factor=1.0;
          //      }

          //      if(scale_factor > 0.0){
          //        scale_factor=Min(1.0,scale_factor);
          //      }

          // Loop over velocity fields.  Only proceed if velocity field mass
          // is nonzero (not numerical noise) and the difference from
          // the centerOfMassVelocity is nonzero (More than one velocity
          // field is contributing to grid vertex).
          for(int n = 0; n < numMatls; n++){
            if(!d_matls.requested(n)) continue;
            Vector deltaVelocity=gvelocity_star[n][c]-centerOfMassVelocity;
            double mass = gmass[n][c];
            if(!compare(mass/centerOfMassMass,0.0)
            && !compare(mass-centerOfMassMass,0.0)){

              // Apply frictional contact IF the surface is in compression
              // OR the surface is stress free and approaching.
              // Otherwise apply free surface conditions (do nothing).
              Vector normal = gsurfnorm[n][c];
              double normalDeltaVel=Dot(deltaVelocity,normal);

              Vector Dv(0.,0.,0.);
              double Tn = normtraction[n][c];
              if((Tn < 0.0) || 
                 (compare(fabs(Tn),0.0) && normalDeltaVel>0.0)){

                // Simplify algorithm in case where approach velocity
                // is in direction of surface normal (no slip).
                Vector normal_normaldV = normal*normalDeltaVel;
                Vector dV_normaldV = deltaVelocity - normal_normaldV;
                if(compare(dV_normaldV.length2(),0.0)){

                  // Calculate velocity change needed to enforce contact
                  Dv=-normal_normaldV;
                }

                // General algorithm, including frictional slip.  The
                // contact velocity change and frictional work are both
                // zero if normalDeltaVel is zero.
                else if(!compare(fabs(normalDeltaVel),0.0)){
                  Vector surfaceTangent= dV_normaldV/dV_normaldV.length();
                  double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                  double frictionCoefficient=
                    Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel));

                  // Calculate velocity change needed to enforce contact
                  Dv= -normal_normaldV
                    -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

                  // Calculate work done by the frictional force (only) if
                  // contact slips.  Because the frictional force opposes motion
                  // it is dissipative and should always be negative per the
                  // conventional definition.  However, here it is calculated
                  // as positive (Work=-force*distance).
                  if(compare(frictionCoefficient,d_mu)){
                    frictionWork[n][c] += mass*frictionCoefficient
                      * (normalDeltaVel*normalDeltaVel) *
                      (tangentDeltaVelocity/fabs(normalDeltaVel)-
                       frictionCoefficient);
                  }
                }

                // Define contact algorithm imposed strain, find maximum
                Vector epsilon=(Dv/dx)*delT;
                double epsilon_max=
                  Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
                epsilon_max_max=max(epsilon_max,epsilon_max_max);
                if(!compare(epsilon_max,0.0)){
                  epsilon_max *=Max(1.0, mass/(centerOfMassMass-mass));

                  // Scale velocity change if contact algorithm imposed strain
                  // is too large.
                  double ff=Min(epsilon_max,.5)/epsilon_max;
                  Dv=Dv*ff;
                }
                Dv=scale_factor*Dv;
                gvelocity_star[n][c]+=Dv;
              } // traction
            }   // if !compare && !compare
          }     // for numMatls
        }      // volume constraint
      }       // if centerofmass > 0
    }         // nodeiterator

    //  print out epsilon_max_max
    //  static int ts=0;
    //  static ofstream tmpout("max_strain.dat");

    //  tmpout << ts << " " << epsilon_max_max << endl;
    //  ts++;

    // This converts frictional work into a temperature rate
    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );

      if(!d_matls.requested(m)) {
        for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
          frictionWork[m][*iter] = 0;
        }  
      } else {
        double c_v = mpm_matl->getSpecificHeat();
        for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          frictionWork[m][c] /= (c_v * gmass[m][c] * delT);
          if(frictionWork[m][c]<0.0){
            cout << "dT/dt is negative: " << frictionWork[m][c] << endl;
          }
        }
      }
    }

  }
}

void FrictionContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                          const PatchSet* patches,
                                                          const MaterialSet* ms)
{
  Task * t = scinew Task("Friction::exMomInterpolated", 
                      this, &FrictionContact::exMomInterpolated);

  Ghost::GhostType  gp;
  int ngc_p;
  d_sharedState->getParticleGhostLayer(gp, ngc_p);

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
  t->computes(lb->gNormTractionLabel);
  t->computes(lb->gSurfNormLabel);
  t->computes(lb->gStressLabel);
  t->modifies(lb->frictionalWorkLabel, mss);
  t->modifies(lb->gVelocityLabel, mss);
  
  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}

void FrictionContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* ms) 
{
  Task * t = scinew Task("Friction::exMomIntegrated", 
                      this, &FrictionContact::exMomIntegrated);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,Ghost::None);
  t->requires(Task::NewDW, lb->gNormTractionLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->gSurfNormLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  t->modifies(             lb->gVelocityStarLabel,  mss);
  t->modifies(             lb->frictionalWorkLabel, mss);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}
