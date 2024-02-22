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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/ClayCurveFit.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <iostream>

using namespace std;
using namespace Uintah;

ClayCurveFit::ClayCurveFit(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  //  For pore collapse part of response
//  ps->require("Ps",        d_initialData.Ps);
  ps->require("Pe",        d_initialData.Pe);
  ps->require("rhoS",      d_initialData.rhoS);
  ps->require("K0",        d_initialData.K0);
  ps->require("Ks",        d_initialData.Ks);
  ps->require("beta",      d_initialData.beta);
  ps->require("minimum_porosity",
                           d_initialData.phi_min);
  //  For M-G part of response
//  ps->require("T_0",       d_initialData.T_0);
//  ps->require("C_0",       d_initialData.C_0);
//  ps->require("Gamma_0",   d_initialData.Gamma_0);
//  ps->require("S_alpha",   d_initialData.S_alpha);
  // For the unloading response
  ps->getWithDefault("Ku", d_initialData.Ku,.1*d_initialData.K0);
  ps->getWithDefault("shear_modulus", d_initialData.shear,      0.0);
  ps->getWithDefault("yield_stress",  d_initialData.FlowStress, 9.e99);

  alphaLabel              = VarLabel::create("p.alpha",
                            ParticleVariable<double>::getTypeDescription());
  alphaMinLabel           = VarLabel::create("p.alphaMin",
                            ParticleVariable<double>::getTypeDescription());
  alphaMinLabel_preReloc  = VarLabel::create("p.alphaMin+",
                            ParticleVariable<double>::getTypeDescription());
  tempAlpha1Label           = VarLabel::create("p.tempAlpha1",
                            ParticleVariable<double>::getTypeDescription());
  tempAlpha1Label_preReloc  = VarLabel::create("p.tempAlpha1+",
                            ParticleVariable<double>::getTypeDescription());
  bElBarLabel               = VarLabel::create("p.bElBar",
                             ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc      = VarLabel::create("p.bElBar+",
                             ParticleVariable<Matrix3>::getTypeDescription());
}

ClayCurveFit::~ClayCurveFit()
{
  VarLabel::destroy(alphaLabel);
  VarLabel::destroy(alphaMinLabel);
  VarLabel::destroy(alphaMinLabel_preReloc);
  VarLabel::destroy(tempAlpha1Label);
  VarLabel::destroy(tempAlpha1Label_preReloc);
  VarLabel::destroy(bElBarLabel);
  VarLabel::destroy(bElBarLabel_preReloc);
}

void ClayCurveFit::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","clay_curve_fit");
  }

//  cm_ps->appendElement("Ps",      d_initialData.Ps);
  cm_ps->appendElement("Pe",      d_initialData.Pe);
  cm_ps->appendElement("rhoS",    d_initialData.rhoS);
  cm_ps->appendElement("K0",      d_initialData.K0);
  cm_ps->appendElement("Ks",      d_initialData.Ks);
  cm_ps->appendElement("Ku",      d_initialData.Ku);
  cm_ps->appendElement("beta",    d_initialData.beta);
  cm_ps->appendElement("minimum_porosity", d_initialData.phi_min);
  cm_ps->appendElement("shear_modulus",    d_initialData.shear);
  cm_ps->appendElement("yield_stress",     d_initialData.FlowStress);
//  cm_ps->appendElement("T_0",     d_initialData.T_0);
//  cm_ps->appendElement("C_0",     d_initialData.C_0);
//  cm_ps->appendElement("Gamma_0", d_initialData.Gamma_0);
//  cm_ps->appendElement("S_alpha", d_initialData.S_alpha);
}

ClayCurveFit* ClayCurveFit::clone()
{
  return scinew ClayCurveFit(*this);
}

void ClayCurveFit::addInitialComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(alphaMinLabel,  matlset);
  task->computes(tempAlpha1Label,matlset);
  task->computes(bElBarLabel,    matlset);
}

void ClayCurveFit::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>      alpha_min;
  ParticleVariable<double>      tAlpha1;
  ParticleVariable<Matrix3> bElBar;
  new_dw->allocateAndPut(alpha_min, alphaMinLabel,   pset);
  new_dw->allocateAndPut(tAlpha1,   tempAlpha1Label, pset);
  new_dw->allocateAndPut(bElBar,    bElBarLabel,     pset);

  double rhoS = d_initialData.rhoS;
  double rho_orig = matl->getInitialDensity();
  double alpha_min0 = rhoS/rho_orig;
  Matrix3 Identity;
  Identity.Identity();

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     alpha_min[*iter]    = alpha_min0;
//     tAlpha1[*iter]      = d_initialData.T_0;
     bElBar[*iter]       = Identity;
  }

  computeStableTimeStep(patch, matl, new_dw);
}

void ClayCurveFit::addParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(alphaMinLabel);
  to.push_back(alphaMinLabel_preReloc);
  from.push_back(tempAlpha1Label);
  to.push_back(tempAlpha1Label_preReloc);
  from.push_back(bElBarLabel);
  to.push_back(bElBarLabel_preReloc);
}

void ClayCurveFit::computeStableTimeStep(const Patch* patch,
                                    const MPMMaterial* matl,
                                    DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,        pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,      pset);
  new_dw->get(pvelocity, lb->pVelocityLabel,    pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double K0 = d_initialData.K0;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     double rhoM = pmass[idx]/pvolume[idx];

     double tmp = K0/rhoM;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt(tmp);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void ClayCurveFit::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    double se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity; Identity.Identity();
    double onethird = (1.0/3.0), sqtwthds = sqrt(2.0/3.0);
    Matrix3 tauDev, tauDevTrial;

    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient, bElBar;
    ParticleVariable<Matrix3> pstress,bElBar_new;
    constParticleVariable<double> alpha_min_old, ptemperature, tempAlpha1_old;
    constParticleVariable<double> pvolume;
    ParticleVariable<double> alpha_min_new, alpha_new, tempAlpha1;
    constParticleVariable<Vector> pvelocity;
    ParticleVariable<double> pdTdt,p_q;
    constParticleVariable<Matrix3>      velGrad;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(pvelocity,                   lb->pVelocityLabel,          pset);
    old_dw->get(ptemperature,                lb->pTemperatureLabel,       pset);
    old_dw->get(deformationGradient,         lb->pDeformationMeasureLabel,pset);
    old_dw->get(alpha_min_old,               alphaMinLabel,               pset);
    old_dw->get(tempAlpha1_old,              tempAlpha1Label,             pset);
    old_dw->get(bElBar,                      bElBarLabel,                 pset);

    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel,              pset);
    new_dw->allocateAndPut(p_q,              lb->p_qLabel_preReloc,       pset);
    new_dw->allocateAndPut(alpha_min_new,     alphaMinLabel_preReloc,     pset);
    new_dw->allocateAndPut(alpha_new,         alphaLabel,                 pset);
    new_dw->allocateAndPut(tempAlpha1,        tempAlpha1Label_preReloc,   pset);
    new_dw->get(pvolume,          lb->pVolumeLabel_preReloc,              pset);
    new_dw->get(velGrad,          lb->pVelGradLabel_preReloc,             pset);
    new_dw->get(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc,  pset);
    new_dw->allocateAndPut(bElBar_new,  bElBarLabel_preReloc,      pset);

    double cv = matl->getSpecificHeat();
    double rho_orig = matl->getInitialDensity();
    double Pe = d_initialData.Pe;
    // Compute alpha0 from material density and rhoS
    double alpha0 = d_initialData.rhoS/rho_orig;
    double K0 = d_initialData.K0;
    double Ks = d_initialData.Ks;
    double Ku = d_initialData.Ku;
    double beta  = d_initialData.beta;
    double shear = d_initialData.shear;
    double rhoS  = d_initialData.rhoS;
    double phi_min  = d_initialData.phi_min;
    double phi_low_lim = phi_min + 0.01;
    double alpha_low_lim = 1./(1. - phi_low_lim);
    double phi_init = 1. - 1./alpha0;
    double rhoL = rhoS/alpha_low_lim;

    // Density and alpha at which model stops being elastic
    double rhoP     = rho_orig/(1.-Pe/K0);
    double alphaP   = rhoS/rhoP;

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
       particleIndex idx = *iter;

      Matrix3 pDefGradInc = deformationGradient_new[idx]
                          * deformationGradient[idx].Inverse();
      double Jinc = pDefGradInc.Determinant();
      double Jnew = deformationGradient_new[idx].Determinant();
      double rhoM = rho_orig/Jnew;  // Current material density

      // This section computes a deviatoric stress if the shear_modulus
      // is non-zero.  This is not standard in a P-Alpha model, but without
      // a deviatoric response, simulations aren't very stable.
      // The deviatoric stress is based on comp_neo_hook type response

      // Get the volume preserving part of the deformation gradient increment
      Matrix3 fBar = pDefGradInc/cbrt(Jinc);

      // Compute the trial elastic part of the volume preserving
      // part of the left Cauchy-Green deformation tensor
      Matrix3 bElBarTrial = fBar*bElBar[idx]*fBar.Transpose();

      double IEl   = onethird*bElBarTrial.Trace();
      double muBar = IEl*shear;

      // tauDevTrial is equal to the shear modulus times dev(bElBar)
      // Compute sTnorm = ||tauDevTrial||
      Matrix3 tauDevTrial = (bElBarTrial - Identity*IEl)*shear;
      double sTnorm      = tauDevTrial.Norm();

      // Check for plastic loading
      double flow = d_initialData.FlowStress;
      double fTrial = sTnorm - sqtwthds*flow;

      if (fTrial > 0.0) {
        // plastic
        // Compute increment of slip in the direction of flow
        double delgamma = fTrial/(2.0*muBar);
        Matrix3 normal   = tauDevTrial/sTnorm;

        // The actual shear stress
        tauDev = tauDevTrial - normal*2.0*muBar*delgamma;

        bElBar_new[idx]     = tauDev/shear + Identity*IEl;
      } else {
        // The actual shear stress
        tauDev          = tauDevTrial;
        bElBar_new[idx] = bElBarTrial;
      }
      // End of the deviatoric stress calculation

      // Begining of the volumetric stress calculation

      // alpha starts at rhoS/rho_orig (>1), drops with compaction
      // alpha = 1 at full density
      // porosity = 1 - 1/alpha
      // alpha = 1/(1-porosity)
      double alpha = rhoS/rhoM;
      alpha_min_new[idx]=min(alpha,alpha_min_old[idx]);
      alpha_new[idx]=alpha;

      double p=0.;
      double dAel_dp=0.;
      double cs=sqrt(Ks/rhoS);
      double ce=sqrt(K0/rho_orig);
      double c = cs; // default to unstressed solid material speed of sound
      double dTdt_plas=0.; //, dTdt_MG=0.;

      if(alpha < alpha0 && alpha > alpha_low_lim){
       if(alpha <= alpha_min_old[idx]){  // loading
        if(alpha <= alpha0 && alpha > alphaP){
          // elastic response
          p = K0*(1.-rho_orig/rhoM);
          c = sqrt(K0/rhoM);
        }
        else if(alpha <= alphaP && alpha > 1.0){
          // crushing out the voids
          double phi = 1. - 1./alpha;
          p = Pe -(1./beta)*log((phi - phi_min)/(phi_init - phi_min));
          c = cs + (ce - cs)*((alpha - 1.)/(alpha0 - 1.));
          dTdt_plas = (-p)*(Jinc-1.)*(1./(rhoM*cv))/delT;
        }
       } else { // alpha < alpha_min, unloading
        if(alpha < alpha0 && alpha >= alphaP && alpha_min_old[idx] >= alphaP){
          // still in initial elastic response
          p = K0*(1.-rho_orig/rhoM);
          c = sqrt(K0/rhoM);
        }
        else if((alpha < alphaP && alpha > alpha_low_lim) 
              || alpha_min_old[idx] < alphaP){
          // First, get plastic pressure

          double phi = 1. - 1./alpha;
          p = Pe -(1./beta)*log((phi - phi_min)/(phi_init - phi_min));

          double h = 1. + (ce - cs)*(alpha - 1.0)/(cs*(alpha0-1.));
          dAel_dp = ((alpha*alpha)/Ks)*(1. - 1./(h*h));
          // Limit the unloading modulus, mostly to avoid numerical issues
          dAel_dp = min(dAel_dp, -1./Ks);
          double dPel = (alpha - alpha_min_old[idx])/dAel_dp;
          p += dPel;
          c = cs + (ce - cs)*((alpha - 1.)/(alpha0 - 1.));
        }
       }
       tempAlpha1[idx] = ptemperature[idx];
      }

      double Ps = Pe - (1./beta)*log((0.01)/(phi_init - phi_min));

      // Response for fully densified solid
      if(alpha <= alpha_low_lim || alpha_min_new[idx] < alpha_low_lim){
        // Get the state data
        p = Ks*(1.-rhoL/rhoM);
        c = sqrt(Ks/rhoM);
        p = Ps + p;

        tempAlpha1[idx] = tempAlpha1_old[idx];
      }

      // Unloading cases that get into either alpha > alpha0, or negative P
      if(alpha > alpha0 || p < 0 ){
          // This still may need some work - Jim (2/11/2011)
          // I think I've improved things, but
          // the plastic work (dTdt_plas) still needs work - Jim (9/8/2011)
          double rho_max = min(rhoS,rhoS/alpha_min_new[idx]);
          p = Ku*(1.-rho_max/rhoM);
      }

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = c;
        double DTrace=velGrad[idx].Trace();
        p_q[idx] = artificialBulkViscosity(DTrace, c_bulk, rhoM, dx_ave);
      } else {
        p_q[idx] = 0.;
      }

      pstress[idx] = Identity*(-p) + tauDev/Jnew;

      // Temp increase
      pdTdt[idx] = dTdt_plas; // + dTdt_MG;

      Vector pvelocity_idx = pvelocity[idx];
      c_dil = c;
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }
  }
}

void ClayCurveFit::addComputesAndRequires(Task* task,
                                     const MPMMaterial* matl,
                                     const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, alphaMinLabel,  matlset, gnone);
  task->requires(Task::OldDW, tempAlpha1Label,matlset, gnone);
  task->requires(Task::OldDW, bElBarLabel,    matlset, gnone);

  task->computes(alphaMinLabel_preReloc,      matlset);
  task->computes(alphaLabel,                  matlset);
  task->computes(tempAlpha1Label_preReloc,    matlset);
  task->computes(bElBarLabel_preReloc,        matlset);
}

void 
ClayCurveFit::addComputesAndRequires(Task* ,
                               const MPMMaterial* ,
                               const PatchSet* ,
                               const bool ) const
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double ClayCurveFit::computeRhoMicroCM(double press, 
                                      const double Temp,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
  cerr << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR ClayCurveFit" << endl;

  return matl->getInitialDensity();
}

void ClayCurveFit::computePressEOSCM(double rhoM,double& pressure, 
                                   double Temp,
                                   double& dp_drho, double& tmp,
                                   const MPMMaterial*, 
                                   double temperature)
{
  cerr << "NO VERSION OF computePressEOSCM EXISTS YET FOR ClayCurveFit" << endl;
  pressure = 101325.;
}

double ClayCurveFit::getCompressibility()
{
  cerr << "NO VERSION OF getCompressibility EXISTS YET FOR ClayCurveFit" << endl;
  return 1.0/d_initialData.K0;
}

namespace Uintah {

} // End namespace Uintah
