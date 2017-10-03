/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/MPM/ConstitutiveModel/JWLppMPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/MinMax.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <iostream>


using namespace std;
using namespace Uintah;

JWLppMPM::JWLppMPM(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_useModifiedEOS = false;

  // Read the ignition pressure
  ps->require("ignition_pressure", d_cm.ignition_pressure);

  // These two parameters are used for the unburned Murnahan EOS
  ps->require("murnaghan_K",    d_cm.K);
  ps->require("murnaghan_n",    d_cm.n);

  // These parameters are used for the product JWL EOS
  ps->require("jwl_A",    d_cm.A);
  ps->require("jwl_B",    d_cm.B);
  ps->require("jwl_C",    d_cm.C);
  ps->require("jwl_R1",   d_cm.R1);
  ps->require("jwl_R2",   d_cm.R2);
  ps->require("jwl_om",   d_cm.omega);
  //ps->require("jwl_rho0", d_cm.rho0);  // Get from matl->getInitialDensity()

  // These parameters are needed for the reaction model
  ps->require("reaction_G",    d_cm.G); // Rate coefficient
  ps->require("reaction_b",    d_cm.b); // Pressure exponent
  // Maximum time increment for burn model subcycling
  ps->getWithDefault("max_burn_timestep_size", d_cm.max_burn_timestep, 1.0e-12);
  // Limit on the fraction that remains unburned
  ps->getWithDefault("max_burned_fraction", d_cm.max_burned_frac, 1.0);

  // Initial stress
  // Fix: Need to make it more general.  Add gravity turn-on option and 
  //      read from file option etc.
  ps->getWithDefault("useInitialStress", d_useInitialStress, false);
  d_init_pressure = 0.0;
  if (d_useInitialStress) {
    ps->getWithDefault("initial_pressure", d_init_pressure, 0.0);
  } 

  // Use subcyling for def grad calculation by default.  Else use
  // Taylor series expansion
  ps->getWithDefault("useTaylorSeriesForDefGrad", d_taylorSeriesForDefGrad, false);
  if (d_taylorSeriesForDefGrad) {
    ps->getWithDefault("num_taylor_terms", d_numTaylorTerms, 10);
  } 

  // Use Newton iterations for stress update by default.  Else use two step
  // algorithm.
  ps->getWithDefault("doFastStressCompute", d_fastCompute, false);
  ps->getWithDefault("tolerance_for_Newton_iterations", d_newtonIterTol, 1.0e-3);
  ps->getWithDefault("max_number_of_Newton_iterations", d_newtonIterMax, 20);

  pProgressFLabel             = VarLabel::create("p.progressF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressFLabel_preReloc    = VarLabel::create("p.progressF+",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel          = VarLabel::create("p.progressdelF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel_preReloc = VarLabel::create("p.progressdelF+",
                               ParticleVariable<double>::getTypeDescription());
}

#if 0
JWLppMPM::JWLppMPM(const JWLppMPM* cm) : ConstitutiveModel(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;

  d_cm.ignition_pressure = cm->d_cm.ignition_pressure;

  d_cm.K = cm->d_cm.K;
  d_cm.n = cm->d_cm.n;

  d_cm.A = cm->d_cm.A;
  d_cm.B = cm->d_cm.B;
  d_cm.C = cm->d_cm.C;
  d_cm.R1 = cm->d_cm.R1;
  d_cm.R2 = cm->d_cm.R2;
  d_cm.omega = cm->d_cm.omega;
  // d_cm.rho0 = cm->d_cm.rho0;

  d_cm.G    = cm->d_cm.G;
  d_cm.b    = cm->d_cm.b;
  d_cm.max_burn_timestep = cm->d_cm.max_burn_timestep;
  d_cm.max_burned_frac = cm->d_cm.max_burned_frac;

  // Initial stress
  d_useInitialStress = cm->d_useInitialStress;
  d_init_pressure = cm->d_init_pressure;

  // Taylor series for deformation gradient calculation
  d_taylorSeriesForDefGrad = cm->d_taylorSeriesForDefGrad;
  d_numTaylorTerms = cm->d_numTaylorTerms;

  // Stress compute algorithms
  d_fastCompute = cm->d_fastCompute;
  d_newtonIterTol = cm->d_newtonIterTol;
  d_newtonIterMax = cm->d_newtonIterMax;
  
  pProgressFLabel          = VarLabel::create("p.progressF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressFLabel_preReloc = VarLabel::create("p.progressF+",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel          = VarLabel::create("p.progressdelF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel_preReloc = VarLabel::create("p.progressdelF+",
                               ParticleVariable<double>::getTypeDescription());
}
#endif

JWLppMPM::~JWLppMPM()
{
  VarLabel::destroy(pProgressFLabel);
  VarLabel::destroy(pProgressFLabel_preReloc);
  VarLabel::destroy(pProgressdelFLabel);
  VarLabel::destroy(pProgressdelFLabel_preReloc);
}

void JWLppMPM::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","jwlpp_mpm");
  }
  
  cm_ps->appendElement("ignition_pressure", d_cm.ignition_pressure);

  cm_ps->appendElement("murnaghan_K", d_cm.K);
  cm_ps->appendElement("murnaghan_n", d_cm.n);

  cm_ps->appendElement("jwl_A",    d_cm.A);
  cm_ps->appendElement("jwl_B",    d_cm.B);
  cm_ps->appendElement("jwl_C",    d_cm.C);
  cm_ps->appendElement("jwl_R1",   d_cm.R1);
  cm_ps->appendElement("jwl_R2",   d_cm.R2);
  cm_ps->appendElement("jwl_om",   d_cm.omega);
  // cm_ps->appendElement("jwl_rho0", d_cm.rho0);

  cm_ps->appendElement("reaction_b",             d_cm.b);
  cm_ps->appendElement("reaction_G",             d_cm.G);
  cm_ps->appendElement("max_burn_timestep_size", d_cm.max_burn_timestep);
  cm_ps->appendElement("max_burned_fraction",    d_cm.max_burned_frac);

  cm_ps->appendElement("useInitialStress", d_useInitialStress);
  if (d_useInitialStress) {
    cm_ps->appendElement("initial_pressure", d_init_pressure);
  }

  cm_ps->appendElement("useTaylorSeriesForDefGrad", d_taylorSeriesForDefGrad);
  if (d_taylorSeriesForDefGrad) {
    cm_ps->appendElement("num_taylor_terms", d_numTaylorTerms);
  }

  cm_ps->appendElement("doFastStressCompute", d_fastCompute);
  cm_ps->appendElement("tolerance_for_Newton_iterations", d_newtonIterTol);
  cm_ps->appendElement("max_number_of_Newton_iterations", d_newtonIterMax);
}

JWLppMPM* JWLppMPM::clone()
{
  return scinew JWLppMPM(*this);
}

void JWLppMPM::initializeCMData(const Patch* patch,
                                const MPMMaterial* matl,
                                DataWarehouse* new_dw)
{
  // Initialize local variables
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<double>  pProgress, pProgressdelF;

  new_dw->allocateAndPut(pProgress,    pProgressFLabel,   pset);
  new_dw->allocateAndPut(pProgressdelF,pProgressdelFLabel,pset);


  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    pProgress[*iter]     = 0.0;
    pProgressdelF[*iter] = 0.0;
  }

  // Initialize the variables shared by all constitutive models
  if (!d_useInitialStress) {
    // This method is defined in the ConstitutiveModel base class.
    initSharedDataForExplicit(patch, matl, new_dw);

  } else {
    // Initial stress option 
    Matrix3 Identity;
    Identity.Identity();
    Matrix3 zero(0.0);

    ParticleVariable<double>  pdTdt;
    ParticleVariable<Matrix3> pDefGrad;
    ParticleVariable<Matrix3> pStress;

    new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel,               pset);
    new_dw->allocateAndPut(pDefGrad,    lb->pDeformationMeasureLabel, pset);
    new_dw->allocateAndPut(pStress,     lb->pStressLabel,             pset);

    // Set the initial pressure
    double p = d_init_pressure;
    Matrix3 sigInit(-p, 0.0, 0.0, 0.0, -p, 0.0, 0.0, 0.0, -p);

    // Compute deformation gradient
    //  using the Murnaghan eos 
    //     p = (1/nK) [J^(-n) - 1]
    //     =>
    //     det(F) = (1 + nKp)^(-1/n)
    //     =>
    //     F_{11} = F_{22} = F_{33} = (1 + nKp)^(-1/3n)
    double F11 = pow((1.0 + d_cm.K*d_cm.n*p), (-1.0/(3.0*d_cm.n)));
    Matrix3 defGrad(F11, 0.0, 0.0, 0.0, F11, 0.0, 0.0, 0.0, F11);

    iter = pset->begin();
    for(;iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pdTdt[idx] = 0.0;
      pStress[idx] = sigInit;
      pDefGrad[idx] = defGrad;
    }
  }

  computeStableTimestep(patch, matl, new_dw);
}

void JWLppMPM::addParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pProgressFLabel);
  to.push_back(pProgressFLabel_preReloc);
  from.push_back(pProgressdelFLabel);
  to.push_back(pProgressdelFLabel_preReloc);
}

void JWLppMPM::computeStableTimestep(const Patch* patch,
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume,ptemperature;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,        lb->pMassLabel,        pset);
  new_dw->get(pvolume,      lb->pVolumeLabel,      pset);
  new_dw->get(pvelocity,    lb->pVelocityLabel,    pset);
  new_dw->get(ptemperature, lb->pTemperatureLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double K    = d_cm.K;
  double n    = d_cm.n;
  //double rho0 = d_cm.rho0;
  double rho0 = matl->getInitialDensity();
  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
     particleIndex idx = *iter;
     // Compute wave speed at each particle, store the maximum
     double rhoM = pmass[idx]/pvolume[idx];
     double dp_drho = (1./(K*rho0))*pow((rhoM/rho0),n-1.);
     c_dil = sqrt(dp_drho);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void JWLppMPM::computeStressTensor(const PatchSubset* patches,
                                   const MPMMaterial* matl,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Constants 
  Vector WaveSpeed(1.e-12, 1.e-12, 1.e-12);
  Matrix3 Identity; Identity.Identity();

  // Material parameters
  double d_K = d_cm.K;
  double d_n = d_cm.n;
  //double d_rho0 = d_cm.rho0; // matl->getInitialDensity();
  double d_rho0 =  matl->getInitialDensity();

  // Loop through patches
  for(int pp=0; pp<patches->size(); pp++){
    const Patch* patch = patches->get(pp);

    double se  = 0.0;
    double c_dil = 0.0;

    // Get data warehouse, particle set, and patch info
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    Vector dx = patch->dCell();

    // variables to hold this timestep's values
    constParticleVariable<double> pmass, pProgressF, pProgressdelF, pvolume_old;
    constParticleVariable<double>       pvolume;
    ParticleVariable<double>      pdTdt, p_q, pProgressF_new, pProgressdelF_new;
    constParticleVariable<Vector>  pvelocity;
    constParticleVariable<Matrix3> pDefGrad, pstress, pDefGrad_new;
    constParticleVariable<Matrix3> velGrad;
    constParticleVariable<int>     pLocalized_old;
    ParticleVariable<Matrix3>      pstress_new;
    ParticleVariable<int>          pLocalized_new;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pvolume_old,         lb->pVolumeLabel,             pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pDefGrad,            lb->pDeformationMeasureLabel, pset);
    old_dw->get(pProgressF,          pProgressFLabel,              pset);
    old_dw->get(pProgressdelF,       pProgressdelFLabel,           pset);
    old_dw->get(pLocalized_old,      lb->pLocalizedMPMLabel,       pset);
    
    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel,               pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,        pset);
    
    new_dw->allocateAndPut(pProgressF_new,    pProgressFLabel_preReloc,   pset);
    new_dw->allocateAndPut(pProgressdelF_new, pProgressdelFLabel_preReloc,pset);
    new_dw->allocateAndPut(pLocalized_new,    lb->pLocalizedMPMLabel_preReloc,   pset);
    
    new_dw->get(pvolume,          lb->pVolumeLabel_preReloc,              pset);
    new_dw->get(velGrad,          lb->pVelGradLabel_preReloc,             pset);
    new_dw->get(pDefGrad_new,     lb->pDeformationMeasureLabel_preReloc,  pset);

    // Compute deformation gradient and velocity gradient at each 
    // particle before pressure stabilization
    for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // If the particle has already failed just ignore
      pLocalized_new[idx] = 0;
      if (pLocalized_old[idx]) {
        pstress_new[idx] = pstress[idx];
        pdTdt[idx] = 0.0;
        p_q[idx] = 0.0;
        
        pProgressF_new[idx]    = pProgressF[idx];
        pProgressdelF_new[idx] = pProgressdelF[idx];
        pLocalized_new[idx]    = pLocalized_old[idx];
        continue;
      }

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

#if 0
      // Improve upon first order estimate of deformation gradient
      if (d_taylorSeriesForDefGrad) {
        // Use Taylor series expansion
        // Compute mid point velocity gradient
        Matrix3 Amat = (pVelGrad[idx] + pVelGrad_new[idx])*(0.5*delT);
        Matrix3 Finc = Amat.Exponential(d_numTaylorTerms);
        Matrix3 Fnew = Finc*pDefGrad[idx];
        pDefGrad_new[idx] = Fnew;
      } else {
        // Use subcycling
        Matrix3 F = pDefGrad[idx];
        double Lnorm_dt = velGrad_new.Norm()*delT;
        int num_subcycles = max(1,2*((int) Lnorm_dt));
        if(num_subcycles > 1000) {
           cout << "NUM_SCS = " << num_subcycles << endl;
        }
        double dtsc = delT/(double (num_subcycles));
        Matrix3 OP_tensorL_DT = Identity + velGrad_new*dtsc;
        for(int n=0; n<num_subcycles; n++){
           F = OP_tensorL_DT*F;
        }
        pDefGrad_new[idx] = F;
      }
#endif
    }

    // Actually compute the updated stress 
    for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
      particleIndex idx = *iter;

      double J = pDefGrad_new[idx].Determinant();
      double rho_cur = d_rho0;

      // If the particle has already failed just ignore
      if (pLocalized_old[idx]){
        continue;
      }

      if (!(J > 0.0)) {
        cerr << "**ERROR in JWL++MPM** Negative Jacobian of deformation gradient" << endl;
        cerr << "idx = " << idx << " J = " << J << " matl = " << matl << endl;
        cerr << "F_old = " << pDefGrad[idx]     << endl;
        cerr << "F_new = " << pDefGrad_new[idx] << endl;
        cerr << "**Particle is being removed from the computation**" << endl;
        //throw InvalidValue("**ERROR**: Error in deformation gradient", __FILE__, __LINE__);

        pstress_new[idx] = Identity*(0.0);
        pdTdt[idx] = 0.0;
        p_q[idx] = 0.0;
        
        pProgressF_new[idx] = pProgressF[idx];
        pProgressdelF_new[idx] = pProgressdelF[idx];
        pLocalized_new[idx] = -999;
        continue;
      }

      // Compute new mass density and update the deformed volume
      rho_cur = d_rho0/J;

      // Update the burn fraction and pressure
      double J_old = pDefGrad[idx].Determinant();
      double p_old = -(1.0/3.0)*pstress[idx].Trace();
      double f_old = pProgressF[idx];
      double f_new = f_old;
      double p_new = p_old;
      computeUpdatedFractionAndPressure(J_old, J, f_old, p_old, delT, f_new, p_new);

      // Update the volume fraction and the stress in the data warehouse
      pProgressdelF_new[idx] = f_new - f_old;
      pProgressF_new[idx] = f_new;
      pstress_new[idx] = Identity*(-p_new);

      Vector pvelocity_idx = pvelocity[idx];

      // Compute wave speed at each particle, store the maximum
      double dp_drho = (1./(d_K*d_rho0))*pow((rho_cur/d_rho0),d_n-1.);
      c_dil = sqrt(dp_drho);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
                                                                                
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(1.0/(d_K*rho_cur));
        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),      lb->StrainEnergyLabel);
    }
  }
}

void JWLppMPM::carryForward(const PatchSubset* patches,
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

void JWLppMPM::addComputesAndRequires(Task* task,
                                      const MPMMaterial* matl,
                                      const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  task->requires(Task::OldDW, lb->pVelGradLabel,      matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pLocalizedMPMLabel, matlset, Ghost::None);
  task->requires(Task::OldDW, pProgressFLabel,        matlset, Ghost::None);
  task->requires(Task::OldDW, pProgressdelFLabel,     matlset, Ghost::None);
  
  task->computes(pProgressFLabel_preReloc,        matlset);
  task->computes(pProgressdelFLabel_preReloc,     matlset);
  task->computes(lb->pLocalizedMPMLabel_preReloc, matlset);
}

void JWLppMPM::addInitialComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet*) const
{ 
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pProgressFLabel,       matlset);
  task->computes(pProgressdelFLabel,    matlset);
}

void 
JWLppMPM::addComputesAndRequires(Task* ,
                                 const MPMMaterial* ,
                                 const PatchSet* ,
                                 const bool ) const
{
}


// This is not yet implemented - JG- 7/26/10
double JWLppMPM::computeRhoMicroCM(double pressure, 
                                   const double p_ref,
                                   const MPMMaterial* matl,
                                   double temperature,
                                   double rho_guess)
{
    cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR JWLppMPM"
       << endl;
    //double rho_orig = d_cm.rho0; //matl->getInitialDensity();
    double rho_orig = matl->getInitialDensity();

    return rho_orig;
}

void JWLppMPM::computePressEOSCM(const double rhoM,double& pressure, 
                                 const double p_ref,
                                 double& dp_drho, double& tmp,
                                 const MPMMaterial* matl,
                                 double temperature)
{
  double A = d_cm.A;
  double B = d_cm.B;
  double R1 = d_cm.R1;
  double R2 = d_cm.R2;
  double omega = d_cm.omega;
  //double rho0 = d_cm.rho0;
  double rho0 = matl->getInitialDensity();
  double cv = matl->getSpecificHeat();
  double V = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = omega*cv*tmp*rhoM;

  pressure = P1 + P2 + P3;

  dp_drho = (R1*rho0*P1 + R2*rho0*P2)/(rhoM*rhoM) + omega*cv*tmp;
}

// This is not yet implemented - JG- 7/26/10
double JWLppMPM::getCompressibility()
{
   cout << "NO VERSION OF getCompressibility EXISTS YET FOR JWLppMPM"<< endl;
  return 1.0;
}

// This is the burn logic used in the reaction model  (more complex versions
//   are available -- see LS-DYNA manual)
//       df/dt = G (1-f) p^b
//       Forward Euler: f_{n+1} = f_n + G*(1-f_n)*p_n^b*delT
//       Backward Euler: f_{n+1} = f_n + G*(1-f_{n+1})*p_n^b*delT
//                       or, f_{n+1} = (f_n + G*p_n^b*delT)/(1 + G*p_n^b*delT)
//       Fourth-order R-K: f_{n+1} = f_n + 1/6(k1 + 2k2 + 2k3 + k4)
//         k1 = G*(1-f_n)*p_n^b*delT
//         k2 = G*(1-f_n-k1/2)*p_n^b*delT
//         k3 = G*(1-f_n-k2/2)*p_n^b*delT
//         k4 = G*(1-f_n-k3)*p_n^b*delT
// (ignition_pressure in previous versions hardcoded to 2.0e8 Pa)
void
JWLppMPM::computeUpdatedFractionAndPressure(const double& J_old,
                                            const double& J,
                                            const double& f_old_orig,
                                            const double& p_old_orig,
                                            const double& delT,
                                            double& f_new,
                                            double& p_new) const
{
  if ((p_old_orig > d_cm.ignition_pressure) && (f_old_orig < d_cm.max_burned_frac))  {

    //cerr << setprecision(10) << scientific
    //     << " p_old = " << p_old_orig << " ignition = " << d_cm.ignition_pressure 
    //     << " f_old = " << f_old_orig << " max_burn = " << d_cm.max_burned_frac 
    //     << " f_old - max_f = " << f_old_orig - d_cm.max_burned_frac << endl;
    int numCycles = max(1, (int) ceil(delT/d_cm.max_burn_timestep));  
    double delTinc = delT/((double)numCycles);
    double delJ = J/J_old;
    double delJinc = pow(delJ, 1.0/((double)numCycles));
    double p_old = p_old_orig;
    double f_old = f_old_orig;
    double J_new = J_old;
    f_new = f_old_orig;
    p_new = p_old_orig;
   
    if (d_fastCompute) {
      //cerr << "Using Fast" << endl;
      for (int ii = 0; ii < numCycles; ++ii) {

        // Compute Murnaghan and JWL pressures
        J_new *= delJinc;
        double pM = computePressureMurnaghan(J_new);
        double pJWL = computePressureJWL(J_new);

        computeWithTwoStageBackwardEuler(J_new, f_old, p_old, delTinc, pM, pJWL, f_new, p_new);
        f_old = f_new;
        p_old = p_new;

      }
    } else {
      //cerr << "Using Newton" << endl;
      for (int ii = 0; ii < numCycles; ++ii) {

        // Compute Murnaghan and JWL pressures
        J_new *= delJinc;
        double pM = computePressureMurnaghan(J_new);
        double pJWL = computePressureJWL(J_new);

        computeWithNewtonIterations(J_new, f_old, p_old, delTinc, pM, pJWL, f_new, p_new);
        f_old = f_new;
        p_old = p_new;

      }
    }
    if (f_new > d_cm.max_burned_frac) {
      f_new = d_cm.max_burned_frac;
      double pM = computePressureMurnaghan(J);
      double pJWL = computePressureJWL(J);
      p_new = pM*(1.0 - f_new) + pJWL*f_new;
    } 
  } else {
    // Compute Murnaghan and JWL pressures
    double pM = computePressureMurnaghan(J);
    double pJWL = computePressureJWL(J);

    //  The following computes a pressure for partially burned particles
    //  as a mixture of Murnaghan and JWL pressures, based on pProgressF
    //  This is as described in Eq. 5 of "JWL++: ..." by Souers, et al.
    f_new = f_old_orig;
    p_new = pM*(1.0 - f_new) + pJWL*f_new;
  }

  return;
}

//  This is the original two stage Backward Euler
void
JWLppMPM::computeWithTwoStageBackwardEuler(const double& J,
                                           const double& f_old,
                                           const double& p_old,
                                           const double& delT,
                                           const double& pM,
                                           const double& pJWL,
                                           double& f_new,
                                           double& p_new) const
{
  double fac = (delT*d_cm.G)*pow(p_old, d_cm.b);

  // Backward Euler
  f_new = (f_old + fac)/(1.0 + fac);

  // Forward Euler
  // f_new = f_old + (1.0 - f_old)*fac;

  // Fourth-order R-K
  // double k1 = (1.0 - f_old)*fac;
  // double k2 = (1.0 - f_old - 0.5*k1)*fac;
  // double k3 = (1.0 - f_old - 0.5*k2)*fac;
  // double k4 = (1.0 - f_old - k3)*fac;
  // f_new = f_old + 1.0/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4);

  //if (f_new < 0.0) f_new = 0.0;
  if (f_new > d_cm.max_burned_frac) f_new = d_cm.max_burned_frac;  // Max burned volume fraction 
          
  //  The following computes a pressure for partially burned particles
  //  as a mixture of Murnaghan and JWL pressures, based on pProgressF
  //  This is as described in Eq. 5 of "JWL++: ..." by Souers, et al.
  p_new = pM*(1.0 - f_new) + pJWL*f_new;

  return;
}

//  This is the Newton iteration with Backward Euler
void
JWLppMPM::computeWithNewtonIterations(const double& J,
                                      const double& f_old,
                                      const double& p_old,
                                      const double& delT,
                                      const double& pM,
                                      const double& pJWL,
                                      double& f_new,
                                      double& p_new) const
{
  // Initialize matrices
  vector<double> G(2);  // The vector [F_n+1 P_n+1]^T = 0
  FastMatrix JacobianG(2,2);

  // Initial values of f and p
  f_new = f_old;
  p_new = p_old;

  // Set iteration controls
  int iter = 0;
  double norm = 0.0;

  // Compute G
  computeG(J, f_old, f_new, p_new, pM, pJWL, delT, G);

  // Do Newton iterations
  FastMatrix Jinv(2,2);
  vector<double> Finc(2);
  do {

    // Compute Jacobian of G
    computeJacobianG(J, f_new, p_new, pM, pJWL, delT, JacobianG);

    // Invert Jacobian of G 
    Jinv.destructiveInvert(JacobianG);

    // Compute increment
    Jinv.multiply(G, Finc);
  
    // Update the variables
    f_new -= Finc[0];
    p_new -= Finc[1];

    // Compute G
    computeG(J, f_old, f_new, p_new, pM, pJWL, delT, G);

    // Compute L2 norm and increment iter
    norm = sqrt(G[0]*G[0] + G[1]*G[1]);
    iter++;

  } while ((norm > d_newtonIterTol) && (iter < d_newtonIterMax));

  if (iter > d_newtonIterMax) {
    cerr << "**JWLppMPM** Newton iterations failed to converge." << endl;
    cerr << "iter = " << iter << " norm = " << norm << " tol = " << d_newtonIterTol
           << " p_new = " << p_new << " f_new = " << f_new 
           << " p_old = " << p_old << " f_old = " << f_old << " J = " << J << endl;
    cerr << " pM = " << pM << " pJWL = " << pJWL 
           << " G = [" << G[0] << "," << G[1] << "]"
           << " JacobianG = [[" << JacobianG(0,0) << "," << JacobianG(0,1) << "],["
           << JacobianG(1,0) << "," << JacobianG(1,1) << "]]" << endl;
    cerr << " Jinv = [[" << Jinv(0,0) << "," << Jinv(0,1) << "],["
           << Jinv(1,0) << "," << Jinv(1,1) << "]]" 
           << " Finc = [" << Finc[0] << "," << Finc[1] << "]" << endl;
  }
  if (std::isnan(p_new) || std::isnan(f_new)) {
    cerr << "iter = " << iter << " norm = " << norm << " tol = " << d_newtonIterTol
           << " p_new = " << p_new << " f_new = " << f_new 
           << " p_old = " << p_old << " f_old = " << f_old << " J = " << J << endl;
    cerr << " pM = " << pM << " pJWL = " << pJWL 
           << " G = [" << G[0] << "," << G[1] << "]"
           << " JacobianG = [[" << JacobianG(0,0) << "," << JacobianG(0,1) << "],["
           << JacobianG(1,0) << "," << JacobianG(1,1) << "]]" << endl;
    cerr << " Jinv = [[" << Jinv(0,0) << "," << Jinv(0,1) << "],["
           << Jinv(1,0) << "," << Jinv(1,1) << "]]" 
           << " Finc = [" << Finc[0] << "," << Finc[1] << "]" << endl;
    throw InvalidValue("**JWLppMPM ERROR**: Nan in p_new/f_new value or no convergence", __FILE__, __LINE__);
  }


  return;
}

//------------------------------------------------------------------
// Compute G
//  G = [F_n+1 P_n+1]^T
//   F_n+1 = 0 = f_n+1 - f_n - G*(1 - f_n+1)*(p_n+1)^b*Delta t    
//   P_n+1 = 0 = p_n+1 - (1 - f_n+1) p_m - f_n+1 p_jwl
//------------------------------------------------------------------
void 
JWLppMPM::computeG(const double& J,
                   const double& f_old, 
                   const double& f_new, 
                   const double& p_new,
                   const double& pM,
                   const double& pJWL,
                   const double& delT,
                   vector<double>& G) const
{
  double dfdt_new = computeBurnRate(f_new, p_new);
  double f_func = f_new - f_old - dfdt_new*delT;
  double p_func = p_new - (1.0 - f_new)*pM - f_new*pJWL;
  G[0] = f_func;
  G[1] = p_func;
  return;
}

//------------------------------------------------------------------
// Compute the Jacobian of G
//  J_G = [[dF_n+1/df_n+1 dF_n+1/dp_n+1];[dP_n+1/df_n+1 dP_n+1/dp_n+1]]
//   F_n+1 = 0 = f_n+1 - f_n - G*(1 - f_n+1)*(p_n+1)^b*Delta t    
//   P_n+1 = 0 = p_n+1 - (1 - f_n+1) p_m - f_n+1 p_jwl
//   dF_n+1/df_n+1 = 1 + G*(p_n+1)^b*Delta t    
//   dF_n+1/dp_n+1 =  b*G*(1 - f_n+1)*(p_n+1)^(b-1)*Delta t    
//   dP_n+1/df_n+1 =  p_m - p_jwl
//   dP_n+1/dp_n+1 = 1
//------------------------------------------------------------------
void 
JWLppMPM::computeJacobianG(const double& J,
                           const double& f_new, 
                           const double& p_new,
                           const double& pM,
                           const double& pJWL,
                           const double& delT,
                           FastMatrix& JacobianG) const
{
  double fac = d_cm.G*pow(p_new, d_cm.b)*delT;
  double dF_df = 1.0 + fac;
  double dF_dp = d_cm.b*(1.0 - f_new)*(fac/p_new);
  double dP_df = pM - pJWL;
  JacobianG(0,0) = dF_df;
  JacobianG(0,1) = dF_dp;
  JacobianG(1,0) = dP_df;
  JacobianG(1,1) = 1.0;
  return;
}

//------------------------------------------------------------------
//  df/dt = G (1-f) p^b
//------------------------------------------------------------------
double
JWLppMPM::computeBurnRate(const double& f,
                          const double& p) const
{
  double dfdt = d_cm.G*(1.0 - f)*pow(p, d_cm.b);
  return dfdt;
}

//------------------------------------------------------------------
//  p_m = (1/nK) [J^(-n) - 1]
//------------------------------------------------------------------
double
JWLppMPM::computePressureMurnaghan(const double& J) const
{
  double pM = (1.0/(d_cm.n*d_cm.K))*(pow(J,-d_cm.n) - 1.0);
  return pM;
}

//------------------------------------------------------------------
// p_jwl = A exp(-R1 J) + B exp(-R2 J) + C J^[-(1+omega)]
//------------------------------------------------------------------
double
JWLppMPM::computePressureJWL(const double& J) const
{
  double one_plus_omega = 1.0 + d_cm.omega;
  double A_e_to_the_R1_rho0_over_rhoM = d_cm.A*exp(-d_cm.R1*J);
  double B_e_to_the_R2_rho0_over_rhoM = d_cm.B*exp(-d_cm.R2*J);
  double C_rho_rat_tothe_one_plus_omega = d_cm.C*pow(J, -one_plus_omega);

  double pJWL = A_e_to_the_R1_rho0_over_rhoM +
                B_e_to_the_R2_rho0_over_rhoM +
                C_rho_rat_tothe_one_plus_omega;
  return pJWL;
}

