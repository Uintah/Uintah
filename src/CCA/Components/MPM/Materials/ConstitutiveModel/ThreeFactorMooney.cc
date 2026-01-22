/*
 * The MIT License
 * ... [Standard License Header] ...
 */

#include <CCA/Components/MPM/Materials/ConstitutiveModel/ThreeFactorMooney.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Level.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace std;
using namespace Uintah;

ThreeFactorMooney::ThreeFactorMooney(ProblemSpecP& ps, MPMFlags* Mflag) 
  : ConstitutiveModel(Mflag)
{
  ps->require("he_constant_1", d_initialData.C1);
  ps->require("he_constant_2", d_initialData.C2);
  ps->require("he_constant_3", d_initialData.C5); 
  ps->require("he_PR",         d_initialData.PR);
}

ThreeFactorMooney::~ThreeFactorMooney() {}

void ThreeFactorMooney::outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","three_factor_mooney");
  }
    
  cm_ps->appendElement("he_constant_1", d_initialData.C1);
  cm_ps->appendElement("he_constant_2", d_initialData.C2);
  cm_ps->appendElement("he_constant_3", d_initialData.C5);
  cm_ps->appendElement("he_PR",         d_initialData.PR);
}

ThreeFactorMooney* ThreeFactorMooney::clone()
{
  return scinew ThreeFactorMooney(*this);
}

void ThreeFactorMooney::initializeCMData(const Patch* patch,
                                         const MPMMaterial* matl,
                                         DataWarehouse* new_dw)
{
  initSharedDataForExplicit(patch, matl, new_dw);
  computeStableTimeStep(patch, matl, new_dw);
}

void ThreeFactorMooney::computeStableTimeStep(const Patch* patch,
                                              const MPMMaterial* matl,
                                              DataWarehouse* new_dw)
{
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double C1 = d_initialData.C1;
  double C2 = d_initialData.C2;
  double PR = d_initialData.PR;

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
     particleIndex idx = *iter;
     double mu = 2.*(C1 + C2);
     double c_dil = sqrt(2.*mu*(1.- PR)*pvolume[idx]/((1.-2.*PR)*pmass[idx]));
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new < 1.e-12 ? DBL_MAX : delT_new), lb->delTLabel, patch->getLevel());
}

void ThreeFactorMooney::computeStressTensor(const PatchSubset* patches,
                                            const MPMMaterial* matl,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    Matrix3 Identity, B;
    Identity.Identity();
    double se = 0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Vector dx = patch->dCell();
    int dwi = matl->getDWIndex();

    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> deformationGradient_new, velGrad;
    constParticleVariable<double> pmass, pvolume;
    constParticleVariable<Vector> pvelocity;
    ParticleVariable<Matrix3> pstress;
    ParticleVariable<double> pdTdt, p_q;

    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    new_dw->get(pvolume,             lb->pVolumeLabel_preReloc,    pset);
    new_dw->get(deformationGradient_new, lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->get(velGrad,             lb->pVelGradLabel_preReloc,   pset);

    new_dw->allocateAndPut(pstress,  lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pdTdt,    lb->pdTdtLabel,               pset);
    new_dw->allocateAndPut(p_q,      lb->p_qLabel_preReloc,        pset);

    double C1 = d_initialData.C1;
    double C2 = d_initialData.C2;
    double C5 = d_initialData.C5; 
    double C3 = .5*C1 + C2;
    double PR = d_initialData.PR;
    double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);

    for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pdTdt[idx] = 0.0;

      B = deformationGradient_new[idx] * deformationGradient_new[idx].Transpose();
      double invar1 = B.Trace();
      double invar2 = 0.5*((invar1*invar1) - (B*B).Trace());
      double J = deformationGradient_new[idx].Determinant();
      double invar3 = J*J;

      double w3 = -2.0*C3/(invar3*invar3*invar3) + 2.0*C4*(invar3 -1.0);
      double quadraticDeriv = 2.0 * C5 * (invar1 - 3.0);
      double C1pi1C2 = C1 + invar1*C2 + quadraticDeriv;

      pstress[idx] = (B*C1pi1C2 - (B*B)*C2 + Identity*(invar3*w3))*2.0/J;

      double rho_cur = pmass[idx]/pvolume[idx];
      double c_dil = sqrt((4.*(C1+C2*invar2)/J + 8.*(2.*C3/(invar3*invar3*invar3)+C4*(2.*invar3-1.)) 
                     - Min((pstress[idx])(0,0),(pstress[idx])(1,1),(pstress[idx])(2,2))/J)/rho_cur);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

      if (flag->d_artificial_viscosity) {
        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), sqrt((4.*(C1+C2*invar2)/J)/rho_cur), rho_cur, (dx.x()+dx.y()+dx.z())/3.0);
      } else p_q[idx] = 0.;

      se += (C1*(invar1-3.0) + C2*(invar2-3.0) + C5*(invar1-3.0)*(invar1-3.0) + 
             C3*(1.0/(invar3*invar3) - 1.0) + C4*(invar3-1.0)*(invar3-1.0))*pvolume[idx]/J;
    } 

    WaveSpeed = dx/WaveSpeed;
    new_dw->put(delt_vartype(WaveSpeed.minComponent() < 1.e-12 ? DBL_MAX : WaveSpeed.minComponent()), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy || flag->d_reductionVars->strainEnergy) 
      new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);
  }
}

void ThreeFactorMooney::carryForward(const PatchSubset* patches, const MPMMaterial* matl, DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    carryForwardSharedData(old_dw->getParticleSubset(matl->getDWIndex(), patch), old_dw, new_dw, matl);
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy || flag->d_reductionVars->strainEnergy) 
      new_dw->put(sum_vartype(0.), lb->StrainEnergyLabel);
  }
}

void ThreeFactorMooney::addParticleState(std::vector<const VarLabel*>& from, std::vector<const VarLabel*>& to) {}
void ThreeFactorMooney::addComputesAndRequires(Task* task, const MPMMaterial* matl, const PatchSet* patches) const 
{ addSharedCRForExplicit(task, matl->thisMaterial(), patches); }
void ThreeFactorMooney::addComputesAndRequires(Task* , const MPMMaterial* , const PatchSet* , const bool ) const {}
double ThreeFactorMooney::computeRhoMicroCM(double, const double, const MPMMaterial*, double, double) { return 0.; }
void ThreeFactorMooney::computePressEOSCM(double, double&, double, double&, double&, const MPMMaterial*, double) {}
double ThreeFactorMooney::getCompressibility() { return 1.0; }