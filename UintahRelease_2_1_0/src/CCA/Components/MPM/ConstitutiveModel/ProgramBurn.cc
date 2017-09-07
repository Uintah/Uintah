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

#include <CCA/Components/MPM/ConstitutiveModel/ProgramBurn.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <iostream>

using namespace std;
using namespace Uintah;


ProgramBurn::ProgramBurn(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_useModifiedEOS = false;

  // These two parameters are used for the unburned Murnahan EOS
  ps->require("K",    d_initialData.d_K);
  ps->require("n",d_initialData.d_n);

  // These parameters are used for the product JWL EOS
  ps->require("A",    d_initialData.d_A);
  ps->require("B",    d_initialData.d_B);
  ps->require("C",    d_initialData.d_C);
  ps->require("R1",   d_initialData.d_R1);
  ps->require("R2",   d_initialData.d_R2);
  ps->require("om",   d_initialData.d_om);
  ps->require("rho0", d_initialData.d_rho0);

  // These parameters are needed for the reaction model
  ps->require("starting_location",  d_initialData.d_start_place);
  ps->require("D",                  d_initialData.d_D); // Detonation velocity
  ps->getWithDefault("direction_if_plane", d_initialData.d_direction,
                                                              Vector(0.,0.,0.));
  ps->getWithDefault("T0", d_initialData.d_T0, 0.0);

  pProgressFLabel          = VarLabel::create("p.progressF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressFLabel_preReloc = VarLabel::create("p.progressF+",
                               ParticleVariable<double>::getTypeDescription());
}

ProgramBurn::~ProgramBurn()
{
  VarLabel::destroy(pProgressFLabel);
  VarLabel::destroy(pProgressFLabel_preReloc);
}

void ProgramBurn::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","program_burn");
  }
  
  cm_ps->appendElement("K",    d_initialData.d_K);
  cm_ps->appendElement("n",    d_initialData.d_n);

  cm_ps->appendElement("A",    d_initialData.d_A);
  cm_ps->appendElement("B",    d_initialData.d_B);
  cm_ps->appendElement("C",    d_initialData.d_C);
  cm_ps->appendElement("R1",   d_initialData.d_R1);
  cm_ps->appendElement("R2",   d_initialData.d_R2);
  cm_ps->appendElement("om",   d_initialData.d_om);
  cm_ps->appendElement("rho0", d_initialData.d_rho0);

  cm_ps->appendElement("starting_location",  d_initialData.d_start_place);
  cm_ps->appendElement("direction_if_plane", d_initialData.d_direction);
  cm_ps->appendElement("D",                  d_initialData.d_D);
  cm_ps->appendElement("T0",                 d_initialData.d_T0);
}

ProgramBurn* ProgramBurn::clone()
{
  return scinew ProgramBurn(*this);
}

void ProgramBurn::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double> pProgress;
  new_dw->allocateAndPut(pProgress,pProgressFLabel,pset);

  for(ParticleSubset::iterator iter=pset->begin();iter != pset->end(); iter++){
    pProgress[*iter] = 0.;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void ProgramBurn::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pProgressFLabel);
  to.push_back(pProgressFLabel_preReloc);
}

void ProgramBurn::computeStableTimestep(const Patch* patch,
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

  double K = d_initialData.d_K;
  double n = d_initialData.d_n;
  double rho0 = d_initialData.d_rho0;
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
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

void ProgramBurn::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
    for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    double p,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity; Identity.Identity();

    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass,pProgressF;
    constParticleVariable<double> pvolume;
    constParticleVariable<Vector> pvelocity;
    ParticleVariable<double> pdTdt,p_q,pProgressF_new;
    constParticleVariable<Matrix3> velGrad;
    constParticleVariable<int> pLocalized;
    ParticleVariable<int>      pLocalized_new;
    constParticleVariable<long64> pParticleID;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(pProgressF,          pProgressFLabel,              pset);
    old_dw->get(pLocalized,          lb->pLocalizedMPMLabel,       pset);
    old_dw->get(pParticleID,         lb->pParticleIDLabel,         pset);
    
    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,       pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel,                  pset);
    new_dw->allocateAndPut(p_q,              lb->p_qLabel_preReloc,           pset);
    new_dw->allocateAndPut(pProgressF_new,    pProgressFLabel_preReloc,       pset);
    new_dw->allocateAndPut(pLocalized_new,   lb->pLocalizedMPMLabel_preReloc, pset);
    new_dw->get(pvolume,          lb->pVolumeLabel_preReloc,                  pset);
    new_dw->get(velGrad,          lb->pVelGradLabel_preReloc,                 pset);
    new_dw->get(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc,  pset);

    double time = d_sharedState->getElapsedSimTime() - d_initialData.d_T0;

    double K = d_initialData.d_K;
    double n = d_initialData.d_n;
    double A = d_initialData.d_A;
    double B = d_initialData.d_B;
    double C = d_initialData.d_C;
    double R1 = d_initialData.d_R1;
    double R2 = d_initialData.d_R2;
    double om = d_initialData.d_om;
    double rho0 = d_initialData.d_rho0; // matl->getInitialDensity();

    double A_d=d_initialData.d_direction.x();
    double B_d=d_initialData.d_direction.y();
    double C_d=d_initialData.d_direction.z();

    double x0=d_initialData.d_start_place.x();
    double y0=d_initialData.d_start_place.y();
    double z0=d_initialData.d_start_place.z();

    double D_d = -A_d*x0 - B_d*y0 - C_d*z0;
    double denom = 1.0;
    double plane = 0.;

    if(d_initialData.d_direction.length() > 0.0){
      plane = 1.0;
      denom = sqrt(A_d*A_d + B_d*B_d + C_d*C_d);
    }

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      pLocalized_new[idx] = pLocalized[idx];

      Point p = px[idx];

      double dist_plane = fabs(A_d*p.x() + B_d*p.y() + C_d*p.z() + D_d)/denom;

      double dist_straight = (p - d_initialData.d_start_place).length();

      double dist = dist_plane*plane + dist_straight*(1.-plane);

      double t_b = dist/d_initialData.d_D;

      double delta_L = 1.5*pow(pmass[idx]/rho0,1./3.)/d_initialData.d_D;

      if (time >= t_b){
        pProgressF_new[idx] = (time - t_b)/delta_L;
        if(pProgressF_new[idx]>0.96){
          pProgressF_new[idx]=1.0;
        }
      }
      else{
        pProgressF_new[idx]=0.0;
      }

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;
    }

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      double J = deformationGradient_new[idx].Determinant();

      if (J<=0.0) {
        double Jold = deformationGradient[idx].Determinant();
        cout<<"negative J in ProgramBurn, J="<<J<<", Jold = " << Jold << endl;
        cout << "pos = " << px[idx] << endl;
        pLocalized_new[idx]=-999;
        cout<< "localizing (deleting) particle "<<pParticleID[idx]<<endl;
        cout<< "material = " << dwi << endl << "Momentum deleted = "
                                    << pvelocity[idx]*pmass[idx] <<endl;
        J=1;
      }

      //  The following computes a pressure for partially burned particles
      //  as a mixture of Murnahan and JWL pressures, based on pProgressF
      //  This is as described in Eq. 5 of "JWL++: ..." by Souers, et al.
      double pM = (1./(n*K))*(pow(J,-n)-1.);
      double pJWL=pM;

      // For computing speed of sound if not yet detonating
      double rho_cur = rho0/J;
      double dp_drho = (1./(K*rho0))*pow((rho_cur/rho0),n-1.);
      if(pProgressF_new[idx] > 0.0){
        double one_plus_omega = 1.+om;
        double inv_rho_rat=J; //rho0/rhoM;
        double rho_rat=1./J;  //rhoM/rho0;
        double A_e_to_the_R1_rho0_over_rhoM=A*exp(-R1*inv_rho_rat);
        double B_e_to_the_R2_rho0_over_rhoM=B*exp(-R2*inv_rho_rat);
        double C_rho_rat_tothe_one_plus_omega=C*pow(rho_rat,one_plus_omega);

        pJWL  = A_e_to_the_R1_rho0_over_rhoM +
                B_e_to_the_R2_rho0_over_rhoM +
                C_rho_rat_tothe_one_plus_omega;

        // For computing speed of sound if detonat(ing/ed)
        double rho0_rhoMsqrd = rho0/(rho_cur*rho_cur);
        dp_drho = R1*rho0_rhoMsqrd*A_e_to_the_R1_rho0_over_rhoM
                + R2*rho0_rhoMsqrd*B_e_to_the_R2_rho0_over_rhoM
                + (one_plus_omega/rho_cur)*C_rho_rat_tothe_one_plus_omega;
      }

      p = pM*(1.0-pProgressF_new[idx]) + pJWL*pProgressF_new[idx];

      // compute the total stress
      pstress[idx] = Identity*(-p);

      Vector pvelocity_idx = pvelocity[idx];

      // Compute wave speed at each particle, store the maximum
      c_dil = sqrt(dp_drho);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
                                                                                
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(1./(K*rho_cur));
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

void ProgramBurn::carryForward(const PatchSubset* patches,
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

void ProgramBurn::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);

  task->requires(Task::OldDW, lb->pParticleIDLabel,   matlset, Ghost::None);
  task->requires(Task::OldDW, pProgressFLabel,        matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pLocalizedMPMLabel, matlset, Ghost::None);
  task->computes(pProgressFLabel_preReloc,            matlset);
  task->computes(lb->pLocalizedMPMLabel_preReloc,     matlset);
}

void ProgramBurn::addInitialComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet*) const
{ 
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pProgressFLabel,       matlset);
}


void 
ProgramBurn::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}


// This is not yet implemented - JG- 7/26/10
double ProgramBurn::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
    cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR ProgramBurn"
       << endl;
    double rho_orig = d_initialData.d_rho0; //matl->getInitialDensity();

    return rho_orig;
}

void ProgramBurn::computePressEOSCM(const double rhoM,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& tmp,
                                    const MPMMaterial* matl,
                                    double temperature)
{
  double A = d_initialData.d_A;
  double B = d_initialData.d_B;
  double R1 = d_initialData.d_R1;
  double R2 = d_initialData.d_R2;
  double om = d_initialData.d_om;
  double rho0 = d_initialData.d_rho0;
  double cv = matl->getSpecificHeat();
  double V = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*cv*tmp*rhoM;

  pressure = P1 + P2 + P3;

  dp_drho = (R1*rho0*P1 + R2*rho0*P2)/(rhoM*rhoM) + om*cv*tmp;
}

// This is not yet implemented - JG- 7/26/10
double ProgramBurn::getCompressibility()
{
   cout << "NO VERSION OF getCompressibility EXISTS YET FOR ProgramBurn"<< endl;
  return 1.0;
}

namespace Uintah {
} // End namespace Uintah
