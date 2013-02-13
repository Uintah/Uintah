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

#include <CCA/Components/MPM/ConstitutiveModel/P_Alpha.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/NodeIterator.h> 
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Uintah;

P_Alpha::P_Alpha(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  //  For P-alpha part of response
  ps->require("Ps",        d_initialData.Ps);
  ps->require("Pe",        d_initialData.Pe);
  ps->require("rhoS",      d_initialData.rhoS);
  // Compute alpha0 from material density and rhoS - Jim 9/8/2011
  //ps->require("alpha0",    d_initialData.alpha0);
  ps->require("K0",        d_initialData.K0);
  ps->require("Ks",        d_initialData.Ks);
  //  For M-G part of response
  ps->require("T_0",       d_initialData.T_0);
  ps->require("C_0",       d_initialData.C_0);
  ps->require("Gamma_0",   d_initialData.Gamma_0);
  ps->require("S_alpha",   d_initialData.S_alpha);
  // For the unloading response
  ps->getWithDefault("Ku", d_initialData.Ku,.1*d_initialData.K0);

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
}

P_Alpha::P_Alpha(const P_Alpha* cm) : ConstitutiveModel(cm)
{
  d_initialData.Ps     = cm->d_initialData.Ps;
  d_initialData.Pe     = cm->d_initialData.Pe;
  d_initialData.rhoS   = cm->d_initialData.rhoS;
  // Compute alpha0 from material density and rhoS - Jim 9/8/2011
  //d_initialData.alpha0 = cm->d_initialData.alpha0;
  d_initialData.K0     = cm->d_initialData.K0;
  d_initialData.Ks     = cm->d_initialData.Ks;
  d_initialData.Ku     = cm->d_initialData.Ku;

  d_initialData.T_0    = cm->d_initialData.T_0;
  d_initialData.C_0    = cm->d_initialData.C_0;
  d_initialData.Gamma_0= cm->d_initialData.Gamma_0;
  d_initialData.S_alpha= cm->d_initialData.S_alpha;
}

P_Alpha::~P_Alpha()
{
  VarLabel::destroy(alphaLabel);
  VarLabel::destroy(alphaMinLabel);
  VarLabel::destroy(alphaMinLabel_preReloc);
  VarLabel::destroy(tempAlpha1Label);
  VarLabel::destroy(tempAlpha1Label_preReloc);
}

void P_Alpha::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","p_alpha");
  }

  cm_ps->appendElement("Ps",      d_initialData.Ps);
  cm_ps->appendElement("Pe",      d_initialData.Pe);
  cm_ps->appendElement("rhoS",    d_initialData.rhoS);
  // Compute alpha0 from material density and rhoS - Jim 9/8/2011
  //cm_ps->appendElement("alpha0",  d_initialData.alpha0);
  cm_ps->appendElement("K0",      d_initialData.K0);
  cm_ps->appendElement("Ks",      d_initialData.Ks);
  cm_ps->appendElement("Ku",      d_initialData.Ku);
  cm_ps->appendElement("T_0",     d_initialData.T_0);
  cm_ps->appendElement("C_0",     d_initialData.C_0);
  cm_ps->appendElement("Gamma_0", d_initialData.Gamma_0);
  cm_ps->appendElement("S_alpha", d_initialData.S_alpha);
}

P_Alpha* P_Alpha::clone()
{
  return scinew P_Alpha(*this);
}

void P_Alpha::addInitialComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(alphaMinLabel,  matlset);
  task->computes(tempAlpha1Label,matlset);
}

void P_Alpha::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>      alpha_min;
  ParticleVariable<double>      tAlpha1;
  new_dw->allocateAndPut(alpha_min, alphaMinLabel, pset);
  new_dw->allocateAndPut(tAlpha1, tempAlpha1Label, pset);

  double rhoS = d_initialData.rhoS;
  double rho_orig = matl->getInitialDensity();
  double alpha_min0 = rhoS/rho_orig;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
  // Compute alpha0 from material density and rhoS - Jim 9/8/2011
     // alpha_min[*iter]    = d_initialData.alpha0;
     alpha_min[*iter]    = alpha_min0;
     tAlpha1[*iter]      = d_initialData.T_0;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void P_Alpha::addParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(alphaMinLabel);
  to.push_back(alphaMinLabel_preReloc);
  from.push_back(tempAlpha1Label);
  to.push_back(tempAlpha1Label_preReloc);
}

void P_Alpha::computeStableTimestep(const Patch* patch,
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

void P_Alpha::computeStressTensor(const PatchSubset* patches,
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

    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
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

    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,     pset);
    new_dw->allocateAndPut(p_q,              lb->p_qLabel_preReloc,       pset);
    new_dw->allocateAndPut(alpha_min_new,     alphaMinLabel_preReloc,     pset);
    new_dw->allocateAndPut(alpha_new,         alphaLabel,                 pset);
    new_dw->allocateAndPut(tempAlpha1,        tempAlpha1Label_preReloc,   pset);
    new_dw->get(pvolume,          lb->pVolumeLabel_preReloc,              pset);
    new_dw->get(velGrad,          lb->pVelGradLabel_preReloc,             pset);
    new_dw->get(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc,  pset);

    double rho_orig = matl->getInitialDensity();
    double Ps = d_initialData.Ps;
    double Pe = d_initialData.Pe;
    // Compute alpha0 from material density and rhoS - Jim 9/8/2011
    // double alpha0 = d_initialData.alpha0;
    double alpha0 = d_initialData.rhoS/rho_orig;
    double K0 = d_initialData.K0;
    double Ks = d_initialData.Ks;
    double Ku = d_initialData.Ku;
    double rhoS = d_initialData.rhoS;

    double cv = matl->getSpecificHeat();
    double rhoP     = rho_orig/(1.-Pe/K0);
    double alphaP   = rhoS/rhoP;

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
       particleIndex idx = *iter;

      double Jold = deformationGradient[idx].Determinant();
      double Jnew = deformationGradient_new[idx].Determinant();
      double Jinc = Jnew/Jold;
      double rhoM = rho_orig/Jnew;

      double alpha = rhoS/rhoM;
      alpha_min_new[idx]=min(alpha,alpha_min_old[idx]);
      alpha_new[idx]=alpha;

      double p=0.;
      double dAel_dp=0.;
      double cs=sqrt(Ks/rhoS);
      double ce=sqrt(K0/rho_orig);
      double c = cs; // default to unstressed solid material speed of sound
      double dTdt_plas=0., dTdt_MG=0.;

      if(alpha < alpha0 && alpha >= 1.0){
       if(alpha <= alpha_min_old[idx]){  // loading
        if(alpha <= alpha0 && alpha > alphaP){
          // elastic response
          p = K0*(1.-rho_orig/rhoM);
          c = sqrt(K0/rhoM);
        }
        else if(alpha <= alphaP && alpha > 1.0){
          // crushing out the voids
          p= Ps - (Ps-Pe)*sqrt((alpha - 1.)/(alphaP - 1.0));
          c = cs + (ce - cs)*((alpha - 1.)/(alpha0 - 1.));
          dTdt_plas = (-p)*(Jinc-1.)*(1./(rhoM*cv))/delT;
        }
       } else { // alpha < alpha_min, unloading
        if(alpha < alpha0 && alpha >= alphaP && alpha_min_old[idx] >= alphaP){
          // still in initial elastic response
          p = K0*(1.-rho_orig/rhoM);
          c = sqrt(K0/rhoM);
        }
        else if((alpha < alphaP && alpha > 1.0) || alpha_min_old[idx] < alphaP){
          // First, get plastic pressure
          p= Ps - (Ps-Pe)*sqrt((alpha - 1.)/(alphaP - 1.0));
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

      // Mie-Gruneisen response for fully densified solid
      if(alpha < 1.0 || alpha_min_new[idx] < 1.0){
        // Get the state data
        double Gamma_0 = d_initialData.Gamma_0; //1.54
        double C_0 = d_initialData.C_0; //4029.
        double S_alpha = d_initialData.S_alpha; //1.237;

        // Calc. zeta
        double zeta = (rhoM/rhoS - 1.0);

        // Calculate internal energy E
        double E = (cv)*(ptemperature[idx] - tempAlpha1_old[idx])*rhoS;

        // Calculate the pressure
        p = Gamma_0*E;
        if (rhoM != rhoS) {
          double numer = rhoS*(C_0*C_0)*(1.0/zeta+
                               (1.0-0.5*Gamma_0));
          double denom = 1.0/zeta - (S_alpha-1.0);
          if (denom == 0.0) {
            cout << "rho_0 = " << rhoS << " zeta = " << zeta
                 << " numer = " << numer << endl;
            denom = 1.0e-5;
          }
           p += numer/(denom*denom);
         }
         p = Ps + p;

         double DTrace=velGrad[idx].Trace();
         dTdt_MG = -ptemperature[idx]*Gamma_0*rhoS*DTrace/rhoM;
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

//      double etime = d_sharedState->getElapsedTime();
//      cout << "12345 " << " " << etime << " " << alpha << " " << ptemperature[idx] << " " << p << endl;

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = c;
        double DTrace=velGrad[idx].Trace();
        p_q[idx] = artificialBulkViscosity(DTrace, c_bulk, rhoM, dx_ave);
      } else {
        p_q[idx] = 0.;
      }

      pstress[idx] = Identity*(-p);

      // Temp increase
      pdTdt[idx] = dTdt_plas + dTdt_MG;

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

void P_Alpha::addComputesAndRequires(Task* task,
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
  task->computes(alphaMinLabel_preReloc,      matlset);
  task->computes(alphaLabel,                  matlset);
  task->requires(Task::OldDW, tempAlpha1Label,  matlset, gnone);
  task->computes(tempAlpha1Label_preReloc,      matlset);
}

void 
P_Alpha::addComputesAndRequires(Task* ,
                               const MPMMaterial* ,
                               const PatchSet* ,
                               const bool ) const
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double P_Alpha::computeRhoMicroCM(double press, 
                                      const double Temp,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
  cerr << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR P_Alpha" << endl;

  return matl->getInitialDensity();
}

void P_Alpha::computePressEOSCM(double rhoM,double& pressure, 
                                   double Temp,
                                   double& dp_drho, double& tmp,
                                   const MPMMaterial*, 
                                   double temperature)
{
  cerr << "NO VERSION OF computePressEOSCM EXISTS YET FOR P_Alpha" << endl;
  pressure = 101325.;
}

double P_Alpha::getCompressibility()
{
  cerr << "NO VERSION OF getCompressibility EXISTS YET FOR P_Alpha" << endl;
  return 1.0/d_initialData.K0;
}

namespace Uintah {

} // End namespace Uintah
