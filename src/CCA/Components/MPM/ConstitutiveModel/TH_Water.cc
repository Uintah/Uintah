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

#include <CCA/Components/MPM/ConstitutiveModel/TH_Water.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace std;
using namespace Uintah;

TH_Water::TH_Water(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{

  d_useModifiedEOS = false;
  ps->require("a", d_ID.d_a);
  ps->require("b", d_ID.d_b);
  ps->require("co",d_ID.d_co);
  ps->require("ko",d_ID.d_ko);
  ps->require("To",d_ID.d_To);
  ps->require("L", d_ID.d_L);
  ps->require("vo",d_ID.d_vo);
  ps->getWithDefault("Pref",d_ID.Pref,101325.);

/*  typical SI values
   a = 2*10^-7          (K/Pa)
   b = 2.6              (J/kgK^2)
   co = 4205.7          (J/kgK)
   ko = 5*10^-10        (1/Pa)
   To = 277             (K)
   L = 8*10^-6          (1/K^2)
   vo = 1.00008*10^-3   (m^3/kg)
*/
}

TH_Water::~TH_Water()
{
}

void TH_Water::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","TH_water");
  }
  
  cm_ps->appendElement("a",   d_ID.d_a);
  cm_ps->appendElement("b",   d_ID.d_b);
  cm_ps->appendElement("co",  d_ID.d_co);
  cm_ps->appendElement("ko",  d_ID.d_ko);
  cm_ps->appendElement("To",  d_ID.d_To);
  cm_ps->appendElement("L",   d_ID.d_co);
  cm_ps->appendElement("vo",  d_ID.d_vo);
  cm_ps->appendElement("Pref",d_ID.Pref);
}

TH_Water* TH_Water::clone()
{
  return scinew TH_Water(*this);
}

void TH_Water::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void TH_Water::computeStableTimestep(const Patch* patch,
                                 const MPMMaterial* matl,
                                 DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume,ptemp;
  constParticleVariable<Vector> pvelocity;
  constParticleVariable<Matrix3> pstress;

  new_dw->get(pmass,     lb->pMassLabel,          pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,        pset);
  new_dw->get(pstress,   lb->pStressLabel,        pset);
  new_dw->get(ptemp,     lb->pTemperatureLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel,      pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double d_a  = d_ID.d_a;
  double d_b  = d_ID.d_b;
  double d_L  = d_ID.d_L;
  double d_co = d_ID.d_co;
  double d_ko = d_ID.d_ko;
  double d_vo = d_ID.d_vo;
  double d_To = d_ID.d_To;
  double a_L     = d_a * d_L;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;
      // Compute wave speed at each particle, store the maximum
      double press=(-1./3.)*pstress[idx].Trace() + d_ID.Pref;
      double rhoM=pmass[idx]/pvolume[idx];
      // dp_drho
      double a_press      = d_a * press;
      double a_press_temp = a_press + ptemp[idx];
      double x = a_press_temp - d_To;
      double y = 1. - d_ko * press + x*x * d_L;
      double numerator    = y * y * d_vo;
      double denominator  = d_ko - 2. * a_L * (a_press_temp - d_To) ;
 
      double dp_drho = numerator/denominator;

      //__________________________________
      //  dp_de
      numerator   = 2. * d_L * (a_press_temp - d_To);
 
      double d1   = d_ko - 2. * a_L * (a_press_temp - d_To);
      double d2   = d_co + d_b * ( d_To - ptemp[idx]) 
                  + 2. * press * d_L * d_vo 
                  * (-a_press + d_To - 2. * ptemp[idx]);
      denominator = (d1 * d2 );

      double dp_de = numerator/denominator;

      double c_2 = dp_drho + dp_de * press/(rhoM * rhoM);

      c_dil=sqrt(c_2);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void TH_Water::computeStressTensor(const PatchSubset* patches,
                                   const MPMMaterial* matl,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    double J,Jold,Jinc;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity; Identity.Identity();

    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    constParticleVariable<Matrix3> velGrad;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolume,ptemp;
    constParticleVariable<Vector> pvelocity;
    ParticleVariable<double> pdTdt,p_q;


    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(ptemp,               lb->pTemperatureLabel,        pset);

    new_dw->allocateAndPut(pstress,  lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pdTdt,    lb->pdTdtLabel,               pset);
    new_dw->allocateAndPut(p_q,      lb->p_qLabel_preReloc,        pset);
    new_dw->get(deformationGradient_new,
                            lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->get(pvolume,             lb->pVolumeLabel_preReloc,    pset);
    new_dw->get(velGrad,             lb->pVelGradLabel_preReloc,   pset);

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      J    = deformationGradient_new[idx].Determinant();
      Jold = deformationGradient[idx].Determinant();
      Jinc = J/Jold;

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      //Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*0.5;
      //Matrix3 DPrime = D - Identity*onethird*D.Trace();
      // Viscous part of the stress
      //Matrix3 Shear = DPrime*(2.*viscosity);

      // Get the deformed volume and current density
      double rhoM = rho_orig/J;

      double d_a  = d_ID.d_a;
      double d_b  = d_ID.d_b;
      double d_L  = d_ID.d_L;
      double d_co = d_ID.d_co;
      double d_ko = d_ID.d_ko;
      double d_vo = d_ID.d_vo;
      double d_To = d_ID.d_To;

      double vo_rhoM = d_vo * rhoM;  // common
      double a_L     = d_a * d_L;

      double term1 = 1./(2. * a_L * d_a * vo_rhoM);
      double term2 = ( d_ko + 2. * a_L * (d_To - ptemp[idx]) ) * vo_rhoM;
      double term3 = d_ko * d_ko * vo_rhoM;
      double term4 = 4.*a_L*(d_a -(d_a + (ptemp[idx] - d_To) * d_ko)* vo_rhoM);

      double press  = term1 * (term2 - sqrt( vo_rhoM * (term3 + term4 ) ) );

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*(-(press-d_ID.Pref));// + Shear;

      // Temp increase due to P*dV work
      pdTdt[idx] = (-press)*(Jinc-1.)*(1./(rhoM*d_co))/delT;

      // Compute speed of sound to use in finding delT

      // dp_drho
      double a_press      = d_a * press;
      double a_press_temp = a_press + ptemp[idx];
      double x = a_press_temp - d_To;
      double y = 1. - d_ko * press + x*x * d_L;
      double numerator    = y * y * d_vo;
      double denominator  = ( d_ko - 2. * a_L * (a_press_temp - d_To) );
 
      double dp_drho = numerator/denominator;

      //__________________________________
      //  dp_de
      numerator   = 2. * d_L * (a_press_temp - d_To);
 
      double d1   = d_ko - 2. * a_L * (a_press_temp - d_To);
      double d2   = d_co + d_b * ( d_To - ptemp[idx]) 
                  + 2. * press * d_L * d_vo*(-a_press + d_To - 2. * ptemp[idx]);
      denominator = (d1 * d2 );

      double dp_de = numerator/denominator;

      double c_2 = dp_drho + dp_de * press/(rhoM * rhoM);

      c_dil=sqrt(c_2);

      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
                                                                                
      // Compute artificial viscosity term
//      if (flag->d_artificial_viscosity) {
//        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
//        double c_bulk = sqrt(bulk/rho_cur);
//        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
//        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
//      } else {
        p_q[idx] = 0.;
//      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
  }
}

void TH_Water::addParticleState(std::vector<const VarLabel*>& ,
                                   std::vector<const VarLabel*>& )
{
  // Add the local particle state data for this constitutive model.
}

void TH_Water::carryForward(const PatchSubset* patches,
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

void TH_Water::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);

}

void 
TH_Water::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}


// The "CM" versions use the pressure-volume relationship of the CNH model
double TH_Water::computeRhoMicroCM(double press, 
                                   const double p_ref,
                                   const MPMMaterial* matl,
                                   double temperature,
                                   double rho_guess)
{
//  double rho_orig = matl->getInitialDensity();
//  double bulk = d_initialData.d_Bulk;
  
  double x = d_ID.d_a * press + temperature - d_ID.d_To;
  return  1./(d_ID.d_vo*(1. - d_ID.d_ko * press + d_ID.d_L * x*x ) );
}

void TH_Water::computePressEOSCM(const double rho_cur,double& pressure, 
                                 const double p_ref,
                                 double& dp_drho, double& tmp,
                                 const MPMMaterial* matl,
                                 double temperature)
{
//  double bulk = d_initialData.d_Bulk;
//  double rho_orig = matl->getInitialDensity();

//  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
//  pressure   = p_ref + p_g;
//  dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
//  tmp        = bulk/rho_cur;  // speed of sound squared
  cerr << "TH_Water::computePressEOSCM() not yet implemented" << endl;
}

double TH_Water::getCompressibility()
{
  cerr << "TH_Water::getCompressibility() not yet implemented" << endl;
  //return 1.0/d_initialData.d_Bulk;
  return 1.0;
}


namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(TH_Water::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    Uintah::MPI::Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    Uintah::MPI::Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(TH_Water::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "TH_Water::StateData", 
                                  true, &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
