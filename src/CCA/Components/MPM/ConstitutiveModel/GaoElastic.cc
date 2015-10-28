/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <CCA/Components/MPM/ConstitutiveModel/GaoElastic.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/YieldConditionFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/StabilityCheckFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/FlowStressModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DamageModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/MPMEquationOfStateFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/ShearModulusModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/MeltingTempModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/SpecificHeatModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DevStressModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/PlasticityState.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DeformationState.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Gaussian.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/SymmMatrix3.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/TangentModulusTensor.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <cmath>
#include <iostream>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>

using namespace std;
using namespace Uintah;

static DebugStream cout_EP("EP",false);
static DebugStream cout_EP1("EP1",false);
static DebugStream CSTi("EPi",false);
static DebugStream CSTir("EPir",false);

GaoElastic::GaoElastic(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag), ImplicitCM()
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  if(flag->d_doScalarDiffusion){
    ps->require("volume_expansion_coeff",d_initialData.vol_exp_coeff);
  }else{
    d_initialData.vol_exp_coeff = 0.0;
    ostringstream warn;
    warn << "RFElasticPlastic:: This Constitutive Model requires the use "
         << "of scalar diffusion." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  d_tol = 1.0e-10;
  ps->get("tolerance",d_tol);

  d_useModifiedEOS = false;
  ps->get("useModifiedEOS",d_useModifiedEOS);

  initializeLocalMPMLabels();
}

GaoElastic::GaoElastic(const GaoElastic* cm) :
  ConstitutiveModel(cm), ImplicitCM(cm)
{
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
  d_initialData.vol_exp_coeff = cm->d_initialData.vol_exp_coeff;

  d_tol = cm->d_tol ;
  
  initializeLocalMPMLabels();
}

GaoElastic::~GaoElastic()
{
  // Destructor 
  VarLabel::destroy(pRotationLabel);
  VarLabel::destroy(pStrainRateLabel);
  VarLabel::destroy(pLocalizedLabel);
  VarLabel::destroy(pEnergyLabel);

  VarLabel::destroy(pRotationLabel_preReloc);
  VarLabel::destroy(pStrainRateLabel_preReloc);
  VarLabel::destroy(pLocalizedLabel_preReloc);
  VarLabel::destroy(pEnergyLabel_preReloc);
}

//______________________________________________________________________
//
void GaoElastic::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","gao_elastic");
  }
  
  cm_ps->appendElement("bulk_modulus",            d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",           d_initialData.Shear);
  cm_ps->appendElement("tolerance",               d_tol);
  if(flag->d_doScalarDiffusion){
    cm_ps->appendElement("volume_expansion_coeff",  d_initialData.vol_exp_coeff);
  }
}


GaoElastic* GaoElastic::clone()
{
  return scinew GaoElastic(*this);
}

//______________________________________________________________________
//
void
GaoElastic::initializeLocalMPMLabels()
{
  pRotationLabel = VarLabel::create("p.rotation",
    ParticleVariable<Matrix3>::getTypeDescription());
  pStrainRateLabel = VarLabel::create("p.strainRate",
    ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel = VarLabel::create("p.localized",
    ParticleVariable<int>::getTypeDescription());
  pEnergyLabel = VarLabel::create("p.energy",
    ParticleVariable<double>::getTypeDescription());

  pRotationLabel_preReloc = VarLabel::create("p.rotation+",
    ParticleVariable<Matrix3>::getTypeDescription());
  pStrainRateLabel_preReloc = VarLabel::create("p.strainRate+",
    ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel_preReloc = VarLabel::create("p.localized+",
    ParticleVariable<int>::getTypeDescription());
  pEnergyLabel_preReloc = VarLabel::create("p.energy+",
    ParticleVariable<double>::getTypeDescription());
}
//______________________________________________________________________
//
void 
GaoElastic::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pRotationLabel);
  from.push_back(pStrainRateLabel);
  from.push_back(pLocalizedLabel);
  from.push_back(pEnergyLabel);

  to.push_back(pRotationLabel_preReloc);
  to.push_back(pStrainRateLabel_preReloc);
  to.push_back(pLocalizedLabel_preReloc);
  to.push_back(pEnergyLabel_preReloc);
}
//______________________________________________________________________
//
void 
GaoElastic::addInitialComputesAndRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(pRotationLabel,      matlset);
  task->computes(pStrainRateLabel,    matlset);
  task->computes(pLocalizedLabel,     matlset);
  task->computes(pEnergyLabel,        matlset);
}
//______________________________________________________________________
//
void 
GaoElastic::initializeCMData(const Patch* patch,
                                 const MPMMaterial* matl,
                                 DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  if (flag->d_integrator == MPMFlags::Implicit) 
    initSharedDataForImplicit(patch, matl, new_dw);
  else {
    initSharedDataForExplicit(patch, matl, new_dw);
    computeStableTimestep(patch, matl, new_dw);
  }

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  //cout << "Initialize CM Data in ElasticPlasticHP" << endl;
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pRotation;
  ParticleVariable<double>  pPlasticStrain, pDamage, pPorosity, 
                            pPlasticStrainRate, pStrainRate, pEnergy;
  ParticleVariable<int>     pLocalized;

  new_dw->allocateAndPut(pRotation,          pRotationLabel, pset);
  new_dw->allocateAndPut(pStrainRate,        pStrainRateLabel, pset);
  new_dw->allocateAndPut(pLocalized,         pLocalizedLabel, pset);
  new_dw->allocateAndPut(pEnergy,            pEnergyLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){

    pRotation[*iter] = one;
    pStrainRate[*iter] = 0.0;
    pLocalized[*iter] = 0;
    pEnergy[*iter] = 0.;
  }
}
//______________________________________________________________________
//
void 
GaoElastic::computeStableTimestep(const Patch* patch,
                                      const MPMMaterial* matl,
                                      DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);

  constParticleVariable<double> pMass, pVolume;
  constParticleVariable<Vector> pVelocity;

  new_dw->get(pMass,     lb->pMassLabel,     pset);
  new_dw->get(pVolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pVelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double shear = d_initialData.Shear;
  double bulk = d_initialData.Bulk;

  ParticleSubset::iterator iter = pset->begin(); 
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    Vector pvelocity_idx = pVelocity[idx];
    if(pMass[idx] > 0){
      c_dil = sqrt((bulk + 4.0*shear/3.0)*pVolume[idx]/pMass[idx]);
    }
    else{
      c_dil = 0.0;
      pvelocity_idx = Vector(0.0,0.0,0.0);
    }
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}
//______________________________________________________________________
//
void 
GaoElastic::addComputesAndRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  if (flag->d_integrator == MPMFlags::Implicit) {
    addSharedCRForImplicitHypo(task, matlset, true);
  } else {
    addSharedCRForHypoExplicit(task, matlset, patches);
  }

  // Other constitutive model and input dependent computes and requires

  task->requires(Task::OldDW, pRotationLabel,         matlset, gnone);
  task->requires(Task::OldDW, pStrainRateLabel,       matlset, gnone);
  task->requires(Task::OldDW, pLocalizedLabel,        matlset, gnone);
  task->requires(Task::OldDW, lb->pParticleIDLabel,   matlset, gnone);
  task->requires(Task::OldDW, pEnergyLabel,           matlset, gnone);

  task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 

  if(flag->d_doScalarDiffusion){
    task->requires(Task::OldDW, lb->pConcPreviousLabel, matlset, gnone); 
    task->requires(Task::OldDW, lb->pConcentrationLabel, matlset, gnone); 
  }

  task->computes(pRotationLabel_preReloc,       matlset);
  task->computes(pStrainRateLabel_preReloc,     matlset);
  task->computes(pLocalizedLabel_preReloc,      matlset);
  task->computes(pEnergyLabel_preReloc,         matlset);
}
//______________________________________________________________________
//
void 
GaoElastic::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{

  // General stuff
  Matrix3 one; one.Identity(); Matrix3 zero(0.0);
  Matrix3 tensorD(0.0);                   // Rate of deformation
  Matrix3 tensorW(0.0);                   // Spin 
  Matrix3 tensorF; tensorF.Identity();    // Deformation gradient
  Matrix3 tensorU; tensorU.Identity();    // Right Cauchy-Green stretch
  Matrix3 tensorR; tensorR.Identity();    // Rotation 
  Matrix3 sigma(0.0);                     // The Cauchy stress
  Matrix3 tensorEta(0.0);                 // Deviatoric part of tensor D
  Matrix3 tensorS(0.0);                   // Devaitoric part of tensor Sig
  Matrix3 tensorF_new; tensorF_new.Identity(); // Deformation gradient

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double vol_coeff = d_initialData.vol_exp_coeff;
  double rho_0 = matl->getInitialDensity();
  double sqrtThreeTwo = sqrt(1.5);
  double sqrtTwoThird = 1.0/sqrtThreeTwo;

  //**** Used for reaction diffusion *******
  double concentration;
  double concentration_pn;
  double conc_rate;
  
//  double totalStrainEnergy = 0.0;

  // Loop thru patches
  for(int patchIndex=0; patchIndex<patches->size(); patchIndex++){
    const Patch* patch = patches->get(patchIndex);

    // Get grid size
    Vector dx = patch->dCell();

    // Get the set of particles
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the particle location,  particle mass, particle volume, etc.
    constParticleVariable<double> pMass;
    constParticleVariable<double> pVolume;
    constParticleVariable<double> pTemperature;
    constParticleVariable<double> pTemp_prenew;
    constParticleVariable<double> pConcentration;
    constParticleVariable<double> pConc_prenew;
    constParticleVariable<Vector> pVelocity;
    constParticleVariable<Matrix3> pDeformGrad;
    constParticleVariable<Matrix3> pStress;

    old_dw->get(pMass,          lb->pMassLabel,               pset);
    old_dw->get(pVolume,        lb->pVolumeLabel,             pset);
    old_dw->get(pTemperature,   lb->pTemperatureLabel,        pset);
    old_dw->get(pTemp_prenew,   lb->pTempPreviousLabel,       pset);
    old_dw->get(pVelocity,      lb->pVelocityLabel,           pset);
    old_dw->get(pStress,        lb->pStressLabel,             pset);
    old_dw->get(pDeformGrad,    lb->pDeformationMeasureLabel, pset);

    if(flag->d_doScalarDiffusion){
      old_dw->get(pConcentration, lb->pConcentrationLabel,      pset);
      old_dw->get(pConc_prenew,   lb->pConcPreviousLabel,       pset);
    }

    constParticleVariable<double> pStrainRate, pPlasticStrainRate, pEnergy;
    constParticleVariable<int> pLocalized;
    constParticleVariable<Matrix3> pRotation;

    old_dw->get(pStrainRate,        pStrainRateLabel,        pset);
    old_dw->get(pEnergy,            pEnergyLabel,            pset);
    old_dw->get(pLocalized,         pLocalizedLabel,         pset);
    old_dw->get(pRotation,          pRotationLabel,          pset);

    // Get the particle IDs, useful in case a simulation goes belly up
    constParticleVariable<long64> pParticleID; 
    old_dw->get(pParticleID, lb->pParticleIDLabel, pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    constParticleVariable<Matrix3> pDeformGrad_new, velGrad;
    constParticleVariable<double> pVolume_deformed;
    new_dw->get(pDeformGrad_new,  lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->get(velGrad,          lb->pVelGradLabel_preReloc,            pset);
    new_dw->get(pVolume_deformed, lb->pVolumeLabel_preReloc,             pset);

    // Create and allocate arrays for storing the updated information
    ParticleVariable<Matrix3> pRotation_new;
    ParticleVariable<double>  pStrainRate_new;
    ParticleVariable<int>     pLocalized_new;
    ParticleVariable<double>  pdTdt, p_q, pEnergy_new;
    ParticleVariable<Matrix3> pStress_new;
    
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);

    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel,                 pset);
    new_dw->allocateAndPut(p_q,   lb->p_qLabel_preReloc,          pset);
    new_dw->allocateAndPut(pEnergy_new, pEnergyLabel_preReloc,    pset);

    //______________________________________________________________________
    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero int. heating by default, modify with appropriate sources
      // This has units (in MKS) of K/s  (i.e. temperature/time)
      pdTdt[idx] = 0.0;

      Matrix3 tensorL=velGrad[idx];

      // Carry forward the pLocalized tag for now, alter below
      pLocalized_new[idx] = pLocalized[idx];

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      double J = pDeformGrad_new[idx].Determinant();
      tensorF_new=pDeformGrad_new[idx];

      if(!(J > 0.) || J > 1.e5){
          cerr << "**ERROR** Negative (or huge) Jacobian of deformation gradient."
               << "  Deleting particle " << pParticleID[idx] << endl;
          cerr << "l = " << tensorL << endl;
          cerr << "F_old = " << pDeformGrad[idx] << endl;
          cerr << "J_old = " << pDeformGrad[idx].Determinant() << endl;
          cerr << "F_new = " << tensorF_new << endl;
          cerr << "J = " << J << endl;
          cerr << "Temp = " << pTemperature[idx] << endl;
          cerr << "DWI = " << matl->getDWIndex() << endl;
          cerr << "L.norm()*dt = " << tensorL.Norm()*delT << endl;
          pLocalized_new[idx]=-999;

          tensorL=zero;
          tensorF_new.Identity();
      }

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J;

      // Get concentrations
      // double temperature = pTemperature[idx];
      if(flag->d_doScalarDiffusion){
        concentration = pConcentration[idx];
        concentration_pn = pConc_prenew[idx];
        conc_rate = (concentration - concentration_pn)/delT;
      }

      // Calculate rate of deformation tensor (D)
      tensorD = (tensorL + tensorL.Transpose())*0.5;

      // Compute polar decomposition of F (F = RU)
      pDeformGrad[idx].polarDecompositionRMB(tensorU, tensorR);

      // Rotate the total rate of deformation tensor back to the 
      // material configuration
      tensorD = (tensorR.Transpose())*(tensorD*tensorR);

      // Remove stress free concentration dependent component
      if(flag->d_doScalarDiffusion){
        tensorD = tensorD - one * vol_coeff * conc_rate;
      }

      // Calculate the deviatoric part of the non-thermal part
      // of the rate of deformation tensor
      double dTrace = tensorD.Trace();
      tensorEta = tensorD - one*(dTrace/3.0);
     
      // Calculate strain rate
      pStrainRate_new[idx] = sqrtTwoThird*tensorD.Norm();

      // Rotate the Cauchy stress back to the 
      // material configuration and calculate the deviatoric part
      sigma = pStress[idx];
      sigma = (tensorR.Transpose())*(sigma*tensorR);

      //double pressure = sigma.Trace()/3.0; 
      //tensorS = sigma - one * pressure;

      double mu_cur = shear;

      // compute the local sound wave speed
      double c_dil = sqrt((bulk + 4.0*mu_cur/3.0)/rho_cur);
      //-----------------------------------------------------------------------
      // Stage 2:
      //-----------------------------------------------------------------------
      // Assume elastic deformation to get a trial deviatoric stress
      // This is simply the previous timestep deviatoric stress plus a
      // deviatoric elastic increment based on the shear modulus supplied by
      // the strength routine in use.
      DeformationState* defState = scinew DeformationState();
      defState->tensorD    = tensorD;
      defState->tensorEta  = tensorEta;
      defState->viscoElasticWorkRate = 0.0;
      
      Matrix3 trialS = tensorS + tensorEta*2*shear*delT;

      // Calculate the equivalent stress
      // this will be removed next, 
      // it should be computed in the flow stress routine
      // the flow stress routines should be passed
      //  the entire stress (not just deviatoric)
//      double equivStress = sqrtThreeTwo*trialS.Norm();

      tensorS = trialS;

      // Calculate the updated hydrostatic stress
      //double p = d_eos->computePressure(matl, state, tensorF_new, tensorD,delT);

      //double p = pressure + 200*(concentration); //-concentration_pn);
      //double p = 0.5*bulk*(J - 1.0/J) - 10*concentration;
      //cout << "pressure: " << pressure << " p: " << p << endl;

      //double Dkk = tensorD.Trace();
      //double dTdt_isentropic = d_eos->computeIsentropicTemperatureRate(
      //                                           temperature,rho_0,rho_cur,Dkk);

      //pdTdt[idx] += dTdt_isentropic;

      //pdTdt[idx] += Tdot_VW;


      //Matrix3 tensorHy = one*p;
   
      // Calculate the total stress
      //sigma = tensorS + tensorHy;
      sigma = sigma + (2*shear*tensorEta + one*bulk*dTrace) * delT;

      //-----------------------------------------------------------------------
      // Stage 4:
      //-----------------------------------------------------------------------
      // Find if the particle has failed/localized
//      bool isLocalized = false;
//      double tepla = 0.0;

      //-----------------------------------------------------------------------
      // Stage 5:
      //-----------------------------------------------------------------------

      // Rotate the stress back to the laboratory coordinates using new R
      // Compute polar decomposition of new F (F = RU)
      tensorF_new.polarDecompositionRMB(tensorU, tensorR);

      sigma = (tensorR*sigma)*(tensorR.Transpose());
      
      // if(idx == 1){
      //   cout << "Pid: " << idx << ", Stress: " << sigma << endl;
      //   cout << "Pid: " << idx << ", Volume: " << pVolume[idx] << endl;
      // }

      // Update the kinematic variables
      pRotation_new[idx] = tensorR;

      // Save the new data
      pStress_new[idx] = sigma;
        
      // Rotate the deformation rate back to the laboratory coordinates
      tensorD = (tensorR*tensorD)*(tensorR.Transpose());

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      delete defState;

    }  // end particle loop

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();

    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
  }
}

//______________________________________________________________________
//

void 
GaoElastic::addComputesAndRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patches,
                                       const bool recurse,
                                       const bool SchedParent) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForImplicitHypo(task, matlset, true, recurse, SchedParent);

  Ghost::GhostType  gnone = Ghost::None;
  if(SchedParent){
    // For subscheduler
    task->requires(Task::ParentOldDW, lb->pTempPreviousLabel,  matlset, gnone); 
    task->requires(Task::ParentOldDW, lb->pTemperatureLabel,   matlset, gnone);
  }else{
    // For scheduleIterate
    task->requires(Task::OldDW, lb->pTempPreviousLabel,  matlset, gnone); 
    task->requires(Task::OldDW, lb->pTemperatureLabel,   matlset, gnone);
  }
}
//______________________________________________________________________
//
//______________________________________________________________________
//
double GaoElastic::computeRhoMicroCM(double pressure,
                                         const double p_ref,
                                         const MPMMaterial* matl, 
                                         double temperature,
                                         double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;

  double p_gauge = pressure - p_ref;
  double rho_cur;

  double p_g_over_bulk = p_gauge/bulk;
  rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
   
  return rho_cur;
}
//______________________________________________________________________
//
void GaoElastic::computePressEOSCM(double rho_cur,double& pressure,
                                       double p_ref,  
                                       double& dp_drho, double& tmp,
                                       const MPMMaterial* matl, 
                                       double temperature)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();
  double inv_rho_orig = 1./rho_orig;

  double p_g = .5*bulk*(rho_cur*inv_rho_orig - rho_orig/rho_cur);
  pressure   = p_ref + p_g;
  dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + inv_rho_orig);
  tmp        = bulk/rho_cur;  // speed of sound squared
}
//__________________________________
//
double GaoElastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}
