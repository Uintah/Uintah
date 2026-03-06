/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/SVK.h>
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
#include "svkVUMAT.cpp"

#include <iostream>

using namespace std;
using namespace Uintah;

// Material Constants are C1, C2 and PR (poisson's ratio).  
// The shear modulus = 2(C1 + C2).

SVK::SVK(ProblemSpecP& ps, MPMFlags* Mflag) 
  : ConstitutiveModel(Mflag)
{
  ps->require("YoungsModulus",d_initialData.E);
  ps->require("PR",d_initialData.PR);
}

SVK::~SVK()
{
}


void SVK::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","comp_mooney_rivlin");
  }
    
  cm_ps->appendElement("YoungsModulus",d_initialData.E);
  cm_ps->appendElement("PR",d_initialData.PR);
}

SVK* SVK::clone()
{
  return scinew SVK(*this);
}

void 
SVK::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  computeStableTimeStep(patch, matl, new_dw);
}

void SVK::computeStableTimeStep(const Patch* patch,
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

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double E  = d_initialData.E;
  double PR = d_initialData.PR;

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed + particle velocity at each particle, 
     // store the maximum
     c_dil = sqrt(E*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  if(delT_new < 1.e-12)
    new_dw->put(delt_vartype(DBL_MAX), lb->delTLabel, patch->getLevel());
  else
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void SVK::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Matrix3 Identity,B;
    Identity.Identity();
    double c_dil = 0.0,se=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Vector dx=patch->dCell();

    int dwi = matl->getDWIndex();

    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> pDeformGrad, pstressOld;
    constParticleVariable<Matrix3> velGrad;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass;
    constParticleVariable<double> pvolume;
    constParticleVariable<Vector> pvelocity;
    ParticleVariable<double> pdTdt, p_q;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pstressOld,          lb->pStressLabel,             pset);
    new_dw->get(pvolume,             lb->pVolumeLabel_preReloc,    pset);
    new_dw->get(pDeformGrad, lb->pDeformationMeasureLabel_preReloc,pset);
    new_dw->get(velGrad,             lb->pVelGradLabel_preReloc,   pset);

    new_dw->allocateAndPut(pstress,  lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pdTdt,    lb->pdTdtLabel,               pset);
    new_dw->allocateAndPut(p_q,      lb->p_qLabel_preReloc,        pset);

    double E  = d_initialData.E;
    double PR = d_initialData.PR;

    double props[] = {E, PR};
    int nblock = 1, ndir = 3, nshr = 3, nstatev = 0, nprops = 2;
    const int& iptr = 0;
    const double& dptr = 0.0;
    Matrix3 tensorU, tensorR;

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      // This is evidently not used by this model
      Matrix3 SO = pstressOld[idx];
      double stressOld[6]={SO(0,0),SO(1,1),SO(2,2), SO(0,1), SO(1,2), SO(2,0)}; 

      // Compute polar decomposition of F (F = RU)
      pDeformGrad[idx].polarDecompositionRMB(tensorU, tensorR);

      double stressNew[6] = {0.0};
      double strainInc[6] = {0.0}; // Not used
      double stretchNew[6] = {tensorU(0,0), tensorU(1,1), tensorU(2,2), 
                              tensorU(0,1), tensorU(1,2), tensorU(0,2)};

      // Call the VUMAT function directly
      vumat(&nblock, ndir, nshr, nstatev, iptr, nprops, iptr, 
            dptr, dptr, dptr, nullptr, nullptr, nullptr, 
            props, nullptr, strainInc, nullptr, nullptr, nullptr, 
            nullptr, nullptr, stressOld, nullptr, nullptr, nullptr, 
            nullptr, stretchNew, nullptr, nullptr, stressNew, nullptr, 
            nullptr, nullptr);

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      pstress[idx] = Matrix3(stressNew[0], stressNew[3], stressNew[5],
                             stressNew[3], stressNew[1], stressNew[4],
                             stressNew[5], stressNew[4], stressNew[2]);

      // Compute wave speed + particle velocity at each particle, 
      // store the maximum
      double rho_cur = pmass[idx]/pvolume[idx];
      c_dil = sqrt(E/rho_cur);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double bulk = d_initialData.E/(3.*(1. -2.*d_initialData.PR));
        double c_bulk = sqrt(bulk/rho_cur);
        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }

      // Compute the strain energy for all the particles
      double e = 0.;  // Fix this

      se += e;
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();

    if(delT_new < 1.e-12)
      new_dw->put(delt_vartype(DBL_MAX), lb->delTLabel);
    else
      new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),      lb->StrainEnergyLabel);
    }
  }
}

void SVK::carryForward(const PatchSubset* patches,
                                    const MPMMaterial* matl,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model 
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

         
void SVK::addParticleState(std::vector<const VarLabel*>& from,
                                        std::vector<const VarLabel*>& to)
{
}

void SVK::addComputesAndRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patches ) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);
}

void 
SVK::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}

double SVK::computeRhoMicroCM(double pressure,
                              const double p_ref,
                              const MPMMaterial* matl,
                              double temperature,
                              double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.E/(3.*(1. -2.*d_initialData.PR));

  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));

  return rho_cur;
}

void SVK::computePressEOSCM(double rho_cur,double& pressure,
                                         double p_ref,
                                         double& dp_drho, double& tmp,
                                         const MPMMaterial* matl, 
                                         double temperature)
{
  double bulk = d_initialData.E/(3.*(1. -2.*d_initialData.PR));
  double rho_orig = matl->getInitialDensity();
  double shear = d_initialData.E/(2.*(1+d_initialData.PR));

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.*shear/3.)/rho_cur;  // speed of sound squared
}

double SVK::getCompressibility()
{
  double bulk = d_initialData.E/(3.*(1. -2.*d_initialData.PR));
  return 1.0/bulk;
}

namespace Uintah {
} // End namespace Uintah
