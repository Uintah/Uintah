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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/Salt.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/NodeIterator.h> 
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Uintah;

Salt::Salt(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  ps->require("G",d_initialData.G);
  ps->require("K",d_initialData.K);
  ps->require("A1",d_initialData.A1);
  ps->require("A2",d_initialData.A2);
  ps->require("n1",d_initialData.n1);
  ps->require("n2",d_initialData.n2);
  ps->require("omega_c",d_initialData.wc);
  ps->require("omega_t",d_initialData.wt);
  ps->require("B1",d_initialData.B1);
  ps->require("B2",d_initialData.B2);
  ps->require("D",d_initialData.D);

  initializeLocalMPMLabels();
}

Salt::~Salt()
{
  VarLabel::destroy(pViscoPlasticStrainLabel);
  VarLabel::destroy(pViscoPlasticStrainLabel_preReloc);
  VarLabel::destroy(pOmegaLabel);
  VarLabel::destroy(pOmegaLabel_preReloc);
}

void Salt::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","salt");
  }

  cm_ps->appendElement("G",d_initialData.G);
  cm_ps->appendElement("K",d_initialData.K);
  cm_ps->appendElement("A1",d_initialData.A1);
  cm_ps->appendElement("A2",d_initialData.A2);
  cm_ps->appendElement("n1",d_initialData.n1);
  cm_ps->appendElement("n2",d_initialData.n2);
  cm_ps->appendElement("omega_c",d_initialData.wc);
  cm_ps->appendElement("omega_t",d_initialData.wt);
  cm_ps->appendElement("B1",d_initialData.B1);
  cm_ps->appendElement("B2",d_initialData.B2);
  cm_ps->appendElement("D",d_initialData.D);
}

Salt* Salt::clone()
{
  return scinew Salt(*this);
}

void Salt::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  // Get the particles in the current patch
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(),patch);

  ParticleVariable<double>  pViscoPlasStrain, pOmega;
  new_dw->allocateAndPut(pViscoPlasStrain,  pViscoPlasticStrainLabel, pset);
  new_dw->allocateAndPut(pOmega,            pOmegaLabel,              pset);
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end();iter++){
    pViscoPlasStrain[*iter] = 0;
    pOmega[*iter] = 0;
  }

  computeStableTimeStep(patch, matl, new_dw);
}

void Salt::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pViscoPlasticStrainLabel);
  from.push_back(pOmegaLabel);
  to.push_back(pViscoPlasticStrainLabel_preReloc);
  to.push_back(pOmegaLabel_preReloc);
}

void Salt::computeStableTimeStep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double G = d_initialData.G;
  double bulk = d_initialData.K;
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void Salt::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  double rho_orig = matl->getInitialDensity();
  for(int p=0;p<patches->size();p++){
    double se = 0.0;
    const Patch* patch = patches->get(p);

    Matrix3 Identity; Identity.Identity();
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);
    double twothird = (2.0/3.0);
    double sqrtThreeHalves = sqrt(1.5);

    Vector dx = patch->dCell();
    //double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass, ptemperature,pvolume_new;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<double> pVisPlasStrainOld, pOmegaOld;
    constParticleVariable<Matrix3> velGrad, pDefGrad_new;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    
    old_dw->get(pstress,            lb->pStressLabel,             pset);
    old_dw->get(pmass,              lb->pMassLabel,               pset);
    old_dw->get(pvelocity,          lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,       lb->pTemperatureLabel,        pset);
    old_dw->get(pVisPlasStrainOld,      pViscoPlasticStrainLabel, pset);
    old_dw->get(pOmegaOld,              pOmegaLabel,              pset);

    new_dw->get(pvolume_new,        lb->pVolumeLabel_preReloc,    pset);
    new_dw->get(velGrad,            lb->pVelGradLabel_preReloc,   pset);
    new_dw->get(pDefGrad_new,       lb->pDeformationMeasureLabel_preReloc,
                                                                  pset);
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<double> pdTdt,p_q,pVisPlasStrainNew,pOmegaNew;

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel,              pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,       pset);
    new_dw->allocateAndPut(pVisPlasStrainNew,
                                      pViscoPlasticStrainLabel_preReloc, pset);
    new_dw->allocateAndPut(pOmegaNew, pOmegaLabel_preReloc,              pset);

    double G    = d_initialData.G;
    double bulk = d_initialData.K;
    double A1 = d_initialData.A1; // units 1/seconds
    double n1 = d_initialData.n1;  
    double A2 = d_initialData.A2;  // units 1/seconds 
    double n2 = d_initialData.n2;
    double wc = d_initialData.wc;
    double wt = d_initialData.wt;
    double B1 = d_initialData.B1;
    double B2 = d_initialData.B2;
    double Dam= d_initialData.D;

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      // including effect of thermal strain
      Matrix3 D = 0.5*(velGrad[idx] + velGrad[idx].Transpose());
      double DTrace = D.Trace();
      Matrix3 DPrime = D - Identity*onethird*DTrace;

      // Compute trial stress
      double P_old = onethird*pstress[idx].Trace();
      Matrix3  tensorS = pstress[idx] - P_old*Identity;
      // Compute trial deviatoric stress based, assuming elastic behavior
//      Matrix3  trialS  = tensorS + DPrime*(2*G*delT);
      Matrix3  trialS  = tensorS;

      Vector  eigVal;
      Matrix3 eigVec;
      Vector  eigVector[3];
//      Matrix3 scratch = pstress[idx];
//      scratch = Matrix3(-100,50,100,50,-200,150,100,150,-150);
//      Matrix3 test(0.);

      // Compute principal stress values and directions (Eq. 2.1)
      trialS.eigen(eigVal, eigVec);

      // Limit damage based on state of stress (Eq. 2.3)
      Vector w(pOmegaOld[idx]);
      for(int i=0;i<3;i++){
        if(eigVal[i] < 0){
          w[i]=min(w[i],wt);
        } else {
          w[i]=min(w[i],wc);
        }
//        cout << "eigVal[" << i << "] = " << eigVal[i] << endl;
//        cout << "w[" << i << "] = " << w[i] << endl;
        eigVector[i] = Vector(eigVec(0,i),eigVec(1,i),eigVec(2,i));
//        test += eigVal[i]*Matrix3(eigVector[i],eigVector[i]);
      }
//      cout << "test = " << test << endl;

      // Compute Kachanov stress per Eq. 2.2
      Matrix3 KachStress = Matrix3(eigVal.x()/(1.-w[0]), 0., 0., 
                                   0., eigVal.y()/(1.-w[1]), 0.,
                                   0., 0., eigVal.z()/(1.-w[2]));

      // Compute the vonMises equivalent shear stress (Eq. 2.7)
      double KachPress = onethird*KachStress.Trace();
      Matrix3 KachDev  = KachStress - KachPress*Identity;
      double vonMisesKach = sqrtThreeHalves*KachDev.Norm()+1.e-100;

      // Compute scalar representation of the viscoplastic strain rate (Eq. 2.9)
      double epsDotBar_vp = A1*pow(vonMisesKach/G, n1) 
                          + A2*pow(vonMisesKach/G, n2);

//      cout << "vonMisesKach = " << vonMisesKach << endl;
//      cout << "epsDotBar_vp = " << epsDotBar_vp << endl;

      // Integrate total scalar viscoplastic strain
      pVisPlasStrainNew[idx] = pVisPlasStrainOld[idx] + epsDotBar_vp*delT;
      double epsDotVec_vp[3];

      // Compute viscoplastic strain based on associated flow (Eq. 2.8)
      epsDotVec_vp[0] = (epsDotBar_vp*twothird/vonMisesKach)
                      * (KachDev(0,0) - 0.5*(KachDev(1,1)+KachDev(2,2)));
      epsDotVec_vp[1] = (epsDotBar_vp*twothird/vonMisesKach)
                      * (KachDev(1,1) - 0.5*(KachDev(0,0)+KachDev(2,2)));
      epsDotVec_vp[2] = (epsDotBar_vp*twothird/vonMisesKach)
                      * (KachDev(2,2) - 0.5*(KachDev(0,0)+KachDev(1,1)));


      Matrix3 epsDotMat_vp(0.);
      for(int i=0;i<3;i++){
//      cout << "epsDotVec_vp[" << i << "] = " << epsDotVec_vp[i] << endl;
        epsDotMat_vp += epsDotVec_vp[i]*Matrix3(eigVector[i],eigVector[i]);
      }
      // This is Eq. 2.10 modified with some personal communication
      // with Ben Reedlun of SNL
      double sigKachDB =  B2*(KachPress - B1/(1. - w.maxComponent() ));

      double wdot = 0.;
      if(vonMisesKach > sigKachDB){
//        wdot = (D/G)*epsDotBar_vp;
        wdot = (Dam/G)*(vonMisesKach - sigKachDB)*epsDotBar_vp;
//        cout << "Evolving damage" << endl;
//        wdot = 0.;
      }

      // Integrate damage for this particle
      pOmegaNew[idx] = pOmegaOld[idx] + wdot*delT;

      // Compute the elastic part of the strain rate (Eq. 2.4)
      Matrix3 epsDotEl = DPrime - epsDotMat_vp;

      // Use elastic part of strain rate to update stress
      pstress_new[idx] = pstress[idx] + 
                         (epsDotEl*2.*G + Identity*bulk*DTrace)*delT;

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      // This is wrong for this model for now
//      double e = (D(0,0)*AvgStress(0,0) +
//                  D(1,1)*AvgStress(1,1) +
//                  D(2,2)*AvgStress(2,2) +
//              2.*(D(0,1)*AvgStress(0,1) +
//                  D(0,2)*AvgStress(0,2) +
//                  D(1,2)*AvgStress(1,2))) * pvolume_new[idx]*delT;

//      se += e;

      // get the volumetric part of the deformation
      double J = pDefGrad_new[idx].Determinant();

      // compute wave speed on the particle
      double rho_cur = rho_orig/J;
      c_dil = sqrt((bulk + 4.*G/3.)/rho_cur);

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
        p_q[idx] = artificialBulkViscosity(DTrace, c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    }
  }
}

void Salt::carryForward(const PatchSubset* patches,
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
      new_dw->put(sum_vartype(0.),   lb->StrainEnergyLabel);
    }
  }
}

void Salt::addInitialComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet* patch) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();

  // Other constitutive model and input dependent computes and requires
  task->computes(pViscoPlasticStrainLabel,  matlset);
  task->computes(pOmegaLabel,  matlset);
}

void Salt::addComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  Ghost::GhostType gnone = Ghost::None;
  task->requires(Task::OldDW, pViscoPlasticStrainLabel,  matlset,gnone);
  task->requires(Task::OldDW, pOmegaLabel,               matlset,gnone);
  task->computes(             pViscoPlasticStrainLabel_preReloc, matlset);
  task->computes(             pOmegaLabel_preReloc,              matlset);
}

void Salt::addComputesAndRequires(Task* ,
                                  const MPMMaterial* ,
                                  const PatchSet* ,
                                  const bool ) const
{
  cout << "NO VERSION OF addComputesAndRequires EXISTS YET FOR Salt"<<endl;
}

double Salt::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                      const MPMMaterial* matl, 
                                      double temperature,
                                      double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  //double p_ref=101325.0;
  double p_gauge = pressure - p_ref;
  double rho_cur;
  //double G = d_initialData.G;
  double bulk = d_initialData.K;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 0
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Salt"
       << endl;
#endif
}

void Salt::computePressEOSCM(double rho_cur, double& pressure,
                                    double p_ref,
                                    double& dp_drho,      double& tmp,
                                    const MPMMaterial* matl, 
                                    double temperature)
{

  //double G = d_initialData.G;
  double bulk = d_initialData.K;
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared

#if 0
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Salt"
       << endl;
#endif
}

double Salt::getCompressibility()
{
  return 1.0/d_initialData.K;
}

// Initialize all labels of the particle variables associated with Salt model
void Salt::initializeLocalMPMLabels()
{

  //pPorePressure
  pViscoPlasticStrainLabel = VarLabel::create("p.VisPlasStrain",
                                ParticleVariable<double>::getTypeDescription());
  pViscoPlasticStrainLabel_preReloc = VarLabel::create("p.VisPlasStrain+",
                                ParticleVariable<double>::getTypeDescription());
  pOmegaLabel = VarLabel::create("p.omega",
                                ParticleVariable<double>::getTypeDescription());
  pOmegaLabel_preReloc = VarLabel::create("p.omega+",
                                ParticleVariable<double>::getTypeDescription());
}

namespace Uintah {

} // End namespace Uintah
