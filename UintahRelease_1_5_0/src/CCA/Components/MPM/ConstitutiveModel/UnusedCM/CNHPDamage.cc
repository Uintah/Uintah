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

#include "CNHPDamage.h"
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/SymmMatrix3.h>
#include <Core/Math/Short27.h> //for Fracture
#include <Core/Grid/Variables/NodeIterator.h> 
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>


using std::cerr;
using namespace Uintah;

CNHPDamage::CNHPDamage(ProblemSpecP& ps, MPMFlags* Mflag)
  :CNHDamage(ps, Mflag)
{
  initializeLocalMPMLabels();
  getPlasticityData(ps);
}

CNHPDamage::CNHPDamage(const CNHPDamage* cm):CNHDamage(cm)
{
  initializeLocalMPMLabels();
  setPlasticityData(cm);
}

CNHPDamage::~CNHPDamage()
{
  // Destructor 
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
}

void CNHPDamage::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","cnhp_damage");
  }

  CNHDamage::outputProblemSpec(cm_ps,false);

  cm_ps->appendElement("yield_stress",d_plastic.FlowStress);
  cm_ps->appendElement("hardening_modulus",d_plastic.K);
}



CNHPDamage* CNHPDamage::clone()
{
  return scinew CNHPDamage(*this);
}

void 
CNHPDamage::initializeLocalMPMLabels()
{
  pPlasticStrainLabel =          VarLabel::create("p.plasticStrain",
                         ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
                         ParticleVariable<double>::getTypeDescription());
}

void 
CNHPDamage::getPlasticityData(ProblemSpecP& ps)
{
  ps->require("yield_stress",d_plastic.FlowStress);
  ps->require("hardening_modulus",d_plastic.K);
}

void 
CNHPDamage::setPlasticityData(const CNHPDamage* cm)
{
  d_plastic.FlowStress = cm->d_plastic.FlowStress;
  d_plastic.K = cm->d_plastic.K;
}

void 
CNHPDamage::addInitialComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  CNHDamage::addInitialComputesAndRequires(task, matl, patches);

  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pPlasticStrainLabel, matlset);
}

void 
CNHPDamage::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  CNHDamage::initializeCMData(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pPlasticStrain;
  new_dw->allocateAndPut(pPlasticStrain, pPlasticStrainLabel,  pset);

  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
    pPlasticStrain[*iter] = 0.0;
  }
}

void 
CNHPDamage::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet* patches) const
{
  CNHDamage::addComputesAndRequires(task, matl, patches);

  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::OldDW, pPlasticStrainLabel, matlset, gnone);
  task->computes(pPlasticStrainLabel_preReloc,     matlset);
}

void 
CNHPDamage::computeStressTensor(const PatchSubset* patches,
                                const MPMMaterial* matl,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  // Constants
  double onethird = (1.0/3.0);
  double sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity; Identity.Identity();
  Ghost::GhostType gac = Ghost::AroundCells;

  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double flowStress  = d_plastic.FlowStress;
  double hardModulus = d_plastic.K;

  int dwi = matl->getDWIndex();
  double rho_0 = matl->getInitialDensity();

  // Get delT
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel, getLevel(patches));

  // Particle and grid data
  constParticleVariable<Short27> pgCode;
  constParticleVariable<int>     pLocalized;
  constParticleVariable<double>  pFailureStrain, pErosion;
  constParticleVariable<double>  pMass, pPlasticStrain;
  constParticleVariable<Point>   pX;
  constParticleVariable<Vector>  pSize, pVelocity;
  constParticleVariable<Matrix3> pDefGrad, pBeBar;
  constNCVariable<Vector>        gVelocity;
  constNCVariable<Vector>        GVelocity; 
  ParticleVariable<int>          pLocalized_new;
  ParticleVariable<double>       pFailureStrain_new;
  ParticleVariable<double>       pVol_new, pdTdt, pPlasticStrain_new, p_q;
  ParticleVariable<Matrix3>      pDefGrad_new, pBeBar_new, pStress_new;
  ParticleVariable<Matrix3>      pDeformRate;
  constParticleVariable<long64>  pParticleID;

  // Local variables 
  double J = 0.0, p = 0.0, IEl = 0.0, U = 0.0, W = 0.0, c_dil=0.0;
  double fTrial = 0.0, muBar = 0.0, delgamma = 0.0, sTnorm = 0.0, Jinc = 0.0;
  Matrix3 velGrad(0.0), tauDev(0.0), defGradInc(0.0);
  Matrix3 beBarTrial(0.0), tauDevTrial(0.0), normal(0.0), relDefGradBar(0.0);
  Matrix3 defGrad(0.0);

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    // Initialize patch variables
    double se = 0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    // Get patch info

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get particle info
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pMass,                    lb->pMassLabel,               pset);
    old_dw->get(pPlasticStrain,           pPlasticStrainLabel,          pset);
    old_dw->get(pX,                       lb->pXLabel,                  pset);
    old_dw->get(pSize,                    lb->pSizeLabel,               pset);
    old_dw->get(pVelocity,                lb->pVelocityLabel,           pset);
    old_dw->get(pDefGrad,                 lb->pDeformationMeasureLabel, pset);
    old_dw->get(pBeBar,                   bElBarLabel,                  pset);
    old_dw->get(pLocalized,               pLocalizedLabel,              pset);
    old_dw->get(pFailureStrain,           pFailureStrainLabel,          pset);
    old_dw->get(pErosion,                 lb->pErosionLabel,            pset);
    old_dw->get(pParticleID,              lb->pParticleIDLabel,         pset);

    // Get Grid info
    new_dw->get(gVelocity,   lb->gVelocityStarLabel, dwi, patch, gac, NGN);
    if (flag->d_fracture) {
      new_dw->get(pgCode,    lb->pgCodeLabel, pset);
      new_dw->get(GVelocity, lb->GVelocityStarLabel, dwi, patch, gac, NGN);
    }
    
    // Allocate space for updated particle variables
    new_dw->allocateAndPut(pVol_new, 
                           lb->pVolumeLabel_preReloc,             pset);
    new_dw->allocateAndPut(pdTdt, 
                           lb->pdTdtLabel_preReloc,               pset);
    new_dw->allocateAndPut(pPlasticStrain_new, 
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDefGrad_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pBeBar_new, 
                           bElBarLabel_preReloc,                  pset);
    new_dw->allocateAndPut(pStress_new,        
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pLocalized_new, 
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pFailureStrain_new, 
                           pFailureStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDeformRate, 
                           pDeformRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(p_q,    lb->p_qLabel_preReloc,         pset);

    // Copy failure strains to new dw
    pFailureStrain_new.copyData(pFailureStrain);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;
      // Initialize velocity gradient
      Matrix3 velGrad(0.0);

      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(pX[idx],ni,d_S,pSize[idx],pDefGrad[idx]);

        short pgFld[27];
        if (flag->d_fracture) {
         for(int k=0; k<27; k++){
           pgFld[k]=pgCode[idx][k];
         }
         computeVelocityGradient(velGrad,ni,d_S,oodx,pgFld,gVelocity,GVelocity);
        } else {
        double erosion = pErosion[idx];
        computeVelocityGradient(velGrad,ni,d_S, oodx, gVelocity, erosion);
        }
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(pX[idx],ni,S,d_S,
                                                                    pSize[idx],pDefGrad[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gVelocity,pX[idx]);
      }

      pDeformRate[idx] = (velGrad + velGrad.Transpose())*0.5;
      
      // 1) Compute the deformation gradient increment using the time_step
      //    velocity gradient (F_n^np1 = dudx * dt + Identity)
      // 2) Update the deformation gradient tensor to its time n+1 value.
      defGradInc = velGrad*delT + Identity;
      Jinc = defGradInc.Determinant();
      defGrad = defGradInc*pDefGrad[idx];
      pDefGrad_new[idx] = defGrad;

      // 1) Get the volumetric part of the deformation
      // 2) Compute the deformed volume and new density
      J = defGrad.Determinant();
      double rho_cur = rho_0/J;
      pVol_new[idx]=pMass[idx]/rho_cur;

      // Get the volume preserving part of the deformation gradient increment
//      relDefGradBar = defGradInc*pow(Jinc, -onethird);
      relDefGradBar = defGradInc/cbrt(Jinc);

      // Compute the trial elastic part of the volume preserving 
      // part of the left Cauchy-Green deformation tensor
      beBarTrial = relDefGradBar*pBeBar[idx]*relDefGradBar.Transpose();
      IEl = onethird*beBarTrial.Trace();
      muBar = IEl*shear;

      // tauDevTrial is equal to the shear modulus times dev(bElBar)
      // Compute ||tauDevTrial||
      tauDevTrial = (beBarTrial - Identity*IEl)*shear;
      sTnorm = tauDevTrial.Norm();

      // Check for plastic loading
      double alpha = pPlasticStrain[idx];
      fTrial = sTnorm - sqtwthds*(hardModulus*alpha + flowStress);

      if (fTrial > 0.0) {

        // plastic
        // Compute increment of slip in the direction of flow
        delgamma = (fTrial/(2.0*muBar))/(1.0 + (hardModulus/(3.0*muBar)));
        normal = tauDevTrial/sTnorm;

        // The actual shear stress
        tauDev = tauDevTrial - normal*2.0*muBar*delgamma;

        // Deal with history variables
        pPlasticStrain_new[idx] = alpha + sqtwthds*delgamma;
        pBeBar_new[idx] = tauDev/shear + Identity*IEl;
      }
      else {

        // The actual shear stress
        tauDev = tauDevTrial;

        // elastic
        pPlasticStrain_new[idx] = alpha;
        pBeBar_new[idx] = beBarTrial;
      }

      // get the hydrostatic part of the stress
      p = 0.5*bulk*(J - 1.0/J);

      // compute the total stress (volumetric + deviatoric)
      pStress_new[idx] = Identity*p + tauDev/J;

      // Modify the stress if particle has failed
      updateFailedParticlesAndModifyStress(defGrad, pFailureStrain[idx], 
                                           pLocalized[idx], pLocalized_new[idx],
                                           pStress_new[idx], pParticleID[idx]);

      // Compute the strain energy for non-localized particles
      if(pLocalized_new[idx] == 0){
        U = .5*bulk*(.5*(J*J - 1.0) - log(J));
        W = .5*shear*(pBeBar_new[idx].Trace() - 3.0);
        double e = (U + W)*pVol_new[idx]/J;
        se += e;
      }

      // Compute the local sound speed
      c_dil = sqrt((bulk + 4.*shear/3.)/rho_cur);

      // Compute wave speed at each particle, store the maximum
      Vector pvel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvel.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
        p_q[idx] = artificialBulkViscosity(pDeformRate[idx].Trace(), c_bulk,
                                           rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    } // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();

    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),      lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

void 
CNHPDamage::computeStressTensorImplicit(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  // Constants
  double onethird = (1.0/3.0);
  double sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity; Identity.Identity();
  Ghost::GhostType gac = Ghost::AroundCells;

  double rho_0 = matl->getInitialDensity();
  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double flowStress  = d_plastic.FlowStress;
  double hardModulus = d_plastic.K;

  int dwi = matl->getDWIndex();

  // Particle and grid data
  constParticleVariable<int>     pLocalized;
  constParticleVariable<double>  pFailureStrain;
  constParticleVariable<double>  pMass, pPlasticStrain;
  constParticleVariable<Point>   pX;
  constParticleVariable<Vector>  pSize;
  constParticleVariable<Matrix3> pDefGrad, pBeBar;
  constNCVariable<Vector>        gDisp;
  ParticleVariable<int>          pLocalized_new;
  ParticleVariable<double>       pFailureStrain_new;
  ParticleVariable<double>       pVol_new, pdTdt, pPlasticStrain_new;
  ParticleVariable<Matrix3>      pDefGrad_new, pBeBar_new, pStress_new;

  // Local variables 
  Matrix3 dispGrad(0.0), tauDev(0.0), defGradInc(0.0);
  Matrix3 beBarTrial(0.0), tauDevTrial(0.0), normal(0.0), relDefGradBar(0.0);
  Matrix3 defGrad(0.0);

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    // Initialize patch variables
    double se = 0.0;

    // Get patch info

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get particle info
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pMass,                    lb->pMassLabel,               pset);
    old_dw->get(pPlasticStrain,           pPlasticStrainLabel,          pset);
    old_dw->get(pX,                       lb->pXLabel,                  pset);
    old_dw->get(pSize,                    lb->pSizeLabel,               pset);
    old_dw->get(pDefGrad,                 lb->pDeformationMeasureLabel, pset);
    old_dw->get(pBeBar,                   bElBarLabel,                  pset);
    old_dw->get(pLocalized,               pLocalizedLabel,              pset);
    old_dw->get(pFailureStrain,           pFailureStrainLabel,          pset);

    // Get Grid info
    new_dw->get(gDisp,   lb->dispNewLabel, dwi, patch, gac, 1);
    
    // Allocate space for updated particle variables
    new_dw->allocateAndPut(pVol_new, 
                           lb->pVolumeDeformedLabel,              pset);
    new_dw->allocateAndPut(pdTdt, 
                           lb->pdTdtLabel_preReloc,   pset);
    new_dw->allocateAndPut(pPlasticStrain_new, 
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDefGrad_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pBeBar_new, 
                           bElBarLabel_preReloc,                  pset);
    new_dw->allocateAndPut(pStress_new,        
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pLocalized_new,
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pFailureStrain_new, 
                           pFailureStrainLabel_preReloc,          pset);

    // Copy failure strains to new dw
    pFailureStrain_new.copyData(pFailureStrain);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Compute the displacement gradient and the deformation gradient
      interpolator->findCellAndShapeDerivatives(pX[idx], ni, d_S, pSize[idx],pDefGrad[idx]);
      computeGrad(dispGrad,ni,d_S,oodx,gDisp);

      defGradInc = dispGrad + Identity;         
      double Jinc = defGradInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      defGrad = defGradInc*pDefGrad[idx];
      double J = defGrad.Determinant();
      if (!(J > 0.0)) {
        cerr << getpid() << " " << idx << " "
             << "**ERROR** Negative Jacobian of deformation gradient" << endl;
        throw ParameterNotFound("**ERROR**:CNHPDamage", __FILE__, __LINE__);
      }
      pDefGrad_new[idx] = defGrad;

      // Compute the deformed volume 
      double rho_cur = rho_0/J;
      pVol_new[idx]=pMass[idx]/rho_cur;

      // Compute trial BeBar
      relDefGradBar = defGradInc/cbrt(Jinc);

      // Compute the trial elastic part of the volume preserving 
      // part of the left Cauchy-Green deformation tensor
      beBarTrial = relDefGradBar*pBeBar[idx]*relDefGradBar.Transpose();
      double IEl = onethird*beBarTrial.Trace();
      double muBar = IEl*shear;

      // tauDevTrial is equal to the shear modulus times dev(bElBar)
      // Compute ||tauDevTrial||
      tauDevTrial = (beBarTrial - Identity*IEl)*shear;
      double sTnorm = tauDevTrial.Norm();

      // Check for plastic loading
      double alpha = pPlasticStrain[idx];
      double fTrial = sTnorm - sqtwthds*(hardModulus*alpha + flowStress);

      if (fTrial > 0.0) {

        // plastic
        // Compute increment of slip in the direction of flow
        double delgamma = (fTrial/(2.0*muBar))/
                          (1.0 + (hardModulus/(3.0*muBar)));
        normal = tauDevTrial/sTnorm;

        // The actual shear stress
        tauDev = tauDevTrial - normal*2.0*muBar*delgamma;

        // Deal with history variables
        pPlasticStrain_new[idx] = alpha + sqtwthds*delgamma;
        pBeBar_new[idx] = tauDev/shear + Identity*IEl;
      }
      else {

        // The actual shear stress
        tauDev = tauDevTrial;

        // elastic
        pPlasticStrain_new[idx] = alpha;
        pBeBar_new[idx] = beBarTrial;
      }

      // get the hydrostatic part of the stress
      double p = 0.5*bulk*(J - 1.0/J);

      // compute the total stress (volumetric + deviatoric)
      pStress_new[idx] = Identity*p + tauDev/J;

      // Modify the stress if particle has failed
      updateFailedParticlesAndModifyStress(defGrad, pFailureStrain[idx], 
                                           pLocalized[idx], pLocalized_new[idx],
                                           pStress_new[idx], idx);

      // Compute the strain energy for non-localized particles
      if(pLocalized_new[idx] == 0){
        double U = .5*bulk*(.5*(J*J - 1.0) - log(J));
        double W = .5*shear*(pBeBar_new[idx].Trace() - 3.0);
        double e = (U + W)*pVol_new[idx]/J;
        se += e;
      }
    }
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

void 
CNHPDamage::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet* patches,
                                   const bool recurse,
                                   const bool SchedParent) const
{
  CNHDamage::addComputesAndRequires(task, matl, patches, recurse,SchedParent);

  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gnone = Ghost::None;
  if(SchedParent){
    task->requires(Task::ParentOldDW, pPlasticStrainLabel, matlset, gnone);
  }else{
    task->requires(Task::OldDW, pPlasticStrainLabel, matlset, gnone);
  }
}

void 
CNHPDamage::computeStressTensor(const PatchSubset* patches,
                                const MPMMaterial* matl,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                Solver* solver,
                                const bool )

{
  // Constants
  double onethird = (1.0/3.0);
  double sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity; Identity.Identity();
  Ghost::GhostType gac = Ghost::AroundCells;

  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double flowStress  = d_plastic.FlowStress;
  double hardModulus = d_plastic.K;
  double rho_orig = matl->getInitialDensity();

  int dwi = matl->getDWIndex();
  DataWarehouse* parent_old_dw = 
    new_dw->getOtherDataWarehouse(Task::ParentOldDW);

  // Particle and grid data
  constParticleVariable<double>  pVol, pPlasticStrain, pmass;
  constParticleVariable<Point>   pX;
  constParticleVariable<Vector>  pSize;
  constParticleVariable<Matrix3> pDefGrad, pBeBar;
  constNCVariable<Vector>        gDisp;
  ParticleVariable<double>       pVol_new, pPlasticStrain_new;
  ParticleVariable<Matrix3>      pDefGrad_new, pBeBar_new, pStress_new;

  // Local variables 
  Matrix3 dispGrad(0.0), defGradInc(0.0), defGrad(0.0), relDefGradBar(0.0);
  Matrix3 beBarTrial(0.0), beBarVolTrial(0.0), beBarDevTrial(0.0); 
  Matrix3 tauDev(0.0), tauDevTrial(0.0), normal(0.0);

  // Local variables
  double D[6][6];
  double B[6][24];
  double Bnl[3][24];
  double Kmatrix[24][24];
  int dof[24];
  double v[576];

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    // Set up array for solver
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    ParticleSubset* pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(pX,             lb->pXLabel,                  pset);
    parent_old_dw->get(pSize,          lb->pSizeLabel,               pset);
    parent_old_dw->get(pmass,          lb->pMassLabel,               pset);
    parent_old_dw->get(pDefGrad,       lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pPlasticStrain, pPlasticStrainLabel,          pset);
    parent_old_dw->get(pBeBar,         bElBarLabel,                  pset);

    // Get Grid info
    old_dw->get(gDisp, lb->dispNewLabel, dwi, patch, gac, 1);
    
    // Allocate space for updated particle variables
    new_dw->allocateAndPut(pStress_new,  lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pVol_new,     lb->pVolumeDeformedLabel,  pset);
    new_dw->allocateTemporary(pDefGrad_new,         pset);
    new_dw->allocateTemporary(pPlasticStrain_new,   pset);
    new_dw->allocateTemporary(pBeBar_new,           pset);

    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Compute the displacement gradient and B matrices
      interpolator->findCellAndShapeDerivatives(pX[idx], ni, d_S, pSize[idx],pDefGrad[idx]);
      
      computeGradAndBmats(dispGrad,ni,d_S, oodx,gDisp,l2g,B, Bnl, dof);

      // Compute the deformation gradient increment using the dispGrad
      // Update the deformation gradient tensor to its time n+1 value.
      defGradInc = dispGrad + Identity;
      double Jinc = defGradInc.Determinant();
      pDefGrad_new[idx] = defGradInc*pDefGrad[idx];
      double J = pDefGrad_new[idx].Determinant();

      // Update the particle volume
      double volold = (pmass[idx]/rho_orig);
      double volnew = volold*J;
      pVol_new[idx] = volnew;

      // Compute trial BeBar
//      relDefGradBar = defGradInc*pow(Jinc, -onethird);
      relDefGradBar = defGradInc/cbrt(Jinc);

      // Compute the trial elastic part of the volume preserving 
      // part of the left Cauchy-Green deformation tensor
      beBarTrial = relDefGradBar*pBeBar[idx]*relDefGradBar.Transpose();
      double trBeBarTrial = onethird*beBarTrial.Trace();
      double muBar = shear*trBeBarTrial;
      beBarVolTrial = Identity*trBeBarTrial;
      beBarDevTrial = beBarTrial - beBarVolTrial; 

      // tauDevTrial is equal to the shear modulus times dev(bElBar)
      // Compute ||tauDevTrial||
      tauDevTrial = beBarDevTrial*shear;
      double tauDevNormTrial = tauDevTrial.Norm();

      // Check for plastic loading
      double alpha = pPlasticStrain[idx];
      double fTrial = tauDevNormTrial - 
                      sqtwthds*(hardModulus*alpha + flowStress);

      double delgamma = 0.0;
      if (fTrial > 0.0) {

        // plastic
        // Compute increment of slip in the direction of flow
        delgamma = (fTrial/(2.0*muBar))/(1.0 + (hardModulus/(3.0*muBar)));
        normal = tauDevTrial/tauDevNormTrial;

        // The actual deviatoric stress
        tauDev = tauDevTrial - normal*(2.0*muBar*delgamma);

        // Deal with history variables
        pPlasticStrain_new[idx] = alpha + sqtwthds*delgamma;
        pBeBar_new[idx] = tauDev/shear + beBarVolTrial;
      }
      else {

        // The actual deviatoric stress
        tauDev = tauDevTrial;

        // elastic
        pPlasticStrain_new[idx] = alpha;
        pBeBar_new[idx] = beBarTrial;
      }

      // get the hydrostatic part of the stress
      double p = 0.5*bulk*(J - 1.0/J);

      // compute the total stress (volumetric + deviatoric)
      pStress_new[idx] = Identity*p + tauDev/J;

      // compute the total stress (volumetric + deviatoric)
      //cout << "p = " << p << " J = " << J << " tdev = " << tauDev << endl;

      // Compute the tangent stiffness matrix
      computeTangentStiffnessMatrix(tauDevTrial, normal, 
                                    muBar, delgamma, J, bulk, D);

      // Print out stuff
      /*
      cout.setf(ios::scientific,ios::floatfield);
      cout.precision(10);
      cout << "B = " << endl;
      for(int kk = 0; kk < 24; kk++) {
        for (int ll = 0; ll < 6; ++ll) {
          cout << B[ll][kk] << " " ;
        }
        cout << endl;
      }
      cout << "Bnl = " << endl;
      for(int kk = 0; kk < 24; kk++) {
        for (int ll = 0; ll < 3; ++ll) {
          cout << Bnl[ll][kk] << " " ;
        }
        cout << endl;
      }
      cout << "D = " << endl;
      for(int kk = 0; kk < 6; kk++) {
        for (int ll = 0; ll < 6; ++ll) {
          cout << D[ll][kk] << " " ;
        }
        cout << endl;
      }
      */

      // Compute K matrix = Kmat + Kgeo
      computeStiffnessMatrix(B, Bnl, D, pStress_new[idx], volold, volnew,
                             Kmatrix);

      // Assemble into global K matrix
      for (int I = 0; I < 24; I++){
        for (int J = 0; J < 24; J++){
          v[24*I+J] = Kmatrix[I][J];
        }
      }
      solver->fillMatrix(24,dof,24,dof,v);

    }  // end of loop over particles
    delete interpolator;
    
  }
  solver->flushMatrix();
}

/*! Compute tangent stiffness matrix */
void 
CNHPDamage::computeTangentStiffnessMatrix(const Matrix3& tauDevTrial, 
                                          const Matrix3& normal,
                                          const double&  mubar,
                                          const double&  delGamma,
                                          const double&  J,
                                          const double&  bulk,
                                          double D[6][6])
{
  double twth = 2.0/3.0;
  double frth = 2.0*twth;

  double C_vol[6][6];
  double C_dev[6][6];

  // Initialize all matrices
  for (int ii = 0; ii < 6; ++ii) {
    for (int jj = 0; jj < 6; ++jj) {
      D[ii][jj] = 0.0;
      C_vol[ii][jj] = 0.0;
      C_dev[ii][jj] = 0.0;
    }
  }

  // Volumetric part of the elastic tangent modulus tensor
  double term2 = J*J*bulk;
  double term1 = bulk - term2;
  C_vol[0][0] = bulk;
  C_vol[0][1] = term2;
  C_vol[0][2] = term2;
  C_vol[1][0] = term2;
  C_vol[1][1] = bulk;
  C_vol[1][2] = term2;
  C_vol[2][0] = term2;
  C_vol[2][1] = term2;
  C_vol[2][2] = bulk;
  C_vol[3][3] = term1;
  C_vol[4][4] = term1;
  C_vol[5][5] = term1;

  // Deviatoric part of the elastic tangent modulus tensor
  double Cbar11 = frth*(mubar - tauDevTrial(0,0));
  double Cbar22 = frth*(mubar - tauDevTrial(1,1));
  double Cbar33 = frth*(mubar - tauDevTrial(2,2));
  double fac    = 2.0*mubar;
  double Cbar12 = -twth*(mubar + tauDevTrial(0,0) + tauDevTrial(1,1));
  double Cbar13 = -twth*(mubar + tauDevTrial(0,0) + tauDevTrial(2,2));
  double Cbar23 = -twth*(mubar + tauDevTrial(1,1) + tauDevTrial(2,2));
  C_dev[0][0] = Cbar11;
  C_dev[0][1] = Cbar12;
  C_dev[0][2] = Cbar13;
  C_dev[1][0] = Cbar12;
  C_dev[1][1] = Cbar22;
  C_dev[1][2] = Cbar23;
  C_dev[2][0] = Cbar13;
  C_dev[2][1] = Cbar23;
  C_dev[2][2] = Cbar33;
  C_dev[3][3] = fac;
  C_dev[4][4] = fac;
  C_dev[5][5] = fac;

  for (int ii = 0; ii < 6; ++ii) {
    for (int jj = 0; jj < 6; ++jj) {
      D[ii][jj] = C_vol[ii][jj] + C_dev[ii][jj];
    }
  }

  // Scaling factors for the plastic part of the tangent modulus
  if (delGamma != 0.0) {
    double beta0 = 1.0/(1.0 + bulk/(3.0*mubar));
    double fac1 = tauDevTrial.Norm()/mubar;
    double beta1 = 2.0*delGamma/fac1;
    double beta2 = (1.0 - beta0)*twth*delGamma*fac1;
    double beta3 = beta0 - beta1 + beta2;
    double beta4 = (beta0 - beta1)*fac1;

    // Compute n x n and n x dev(n^2)
    SymmMatrix3 norm(normal);
    SymmMatrix3 nsq = norm.Square();
    SymmMatrix3 nsqDev = nsq.Deviatoric();
    double nn[6][6], ndevnsq[6][6], symndevnsq[6][6];
    norm.Dyad(norm, nn);
    norm.Dyad(nsqDev, ndevnsq);

    // Symmetric part of ndevnsq
    for (int ii = 0; ii < 6; ++ii) {
      for (int jj = 0; jj < 6; ++jj) {
        symndevnsq[ii][jj] = 0.5*(ndevnsq[ii][jj] + ndevnsq[jj][ii]);
      }
    }
    
    // Form the C matrix for the deviatoric part of deformation
    double fac2 = fac*beta3;
    double fac3 = fac*beta4; 
    for (int ii=0; ii < 6; ++ii) {
      for (int jj=0; jj < 6; ++jj) {
        D[ii][jj] -= beta1*C_dev[ii][jj] + fac2*nn[ii][jj] +
                        fac3*symndevnsq[ii][jj];
      }
    }
  }

  // Transform D matrix into form assumed by ImpMPM
  // 11, 22, 33, 12, 13, 23 ???
  for (int ii = 0; ii < 6; ++ii) {
    double temp = D[ii][3];
    D[ii][3] = D[ii][5];
    D[ii][5] = temp;
  }
  for (int ii = 0; ii < 6; ++ii) {
    double temp = D[3][ii];
    D[3][ii] = D[5][ii];
    D[5][ii] = temp;
  }
}

void 
CNHPDamage::carryForward(const PatchSubset* patches,
                         const MPMMaterial* matl,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw)
{
  CNHDamage::carryForward(patches, matl, old_dw, new_dw);

  // Carry forward the data local to this constitutive model 
  int dwi = matl->getDWIndex();
  constParticleVariable<double> pPlasticStrain;
  ParticleVariable<double>      pPlasticStrain_new;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pPlasticStrain, pPlasticStrainLabel,     pset);
    new_dw->allocateAndPut(pPlasticStrain_new, 
                           pPlasticStrainLabel_preReloc, pset);

    pPlasticStrain_new.copyData(pPlasticStrain);
  }
}
 
void 
CNHPDamage::allocateCMDataAddRequires(Task* task,
                                      const MPMMaterial* matl,
                                      const PatchSet* patches,
                                      MPMLabel* lb) const
{
  CNHDamage::allocateCMDataAddRequires(task, matl, patches, lb);

  // Add requires local to this model
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pPlasticStrainLabel_preReloc, matlset, 
                 Ghost::None);
}


void 
CNHPDamage::allocateCMDataAdd(DataWarehouse* new_dw,
                              ParticleSubset* addset,
                              map<const VarLabel*, 
                              ParticleVariableBase*>* newState,
                              ParticleSubset* delset,
                              DataWarehouse* old_dw)
{
  CNHDamage::allocateCMDataAdd(new_dw, addset, newState, delset, old_dw);
  
  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
  constParticleVariable<double>  o_pPlasticStrain;
  new_dw->get(o_pPlasticStrain, pPlasticStrainLabel_preReloc, delset);

  ParticleVariable<double>  pPlasticStrain;
  new_dw->allocateTemporary(pPlasticStrain, addset);

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pPlasticStrain[*n] = o_pPlasticStrain[*o];
  }
  (*newState)[pPlasticStrainLabel] = pPlasticStrain.clone();
}

void 
CNHPDamage::addParticleState(std::vector<const VarLabel*>& from,
                             std::vector<const VarLabel*>& to)
{
  CNHDamage::addParticleState(from, to);

  // Add the local particle state data for this constitutive model.
  from.push_back(pPlasticStrainLabel);
  to.push_back(pPlasticStrainLabel_preReloc);
}

