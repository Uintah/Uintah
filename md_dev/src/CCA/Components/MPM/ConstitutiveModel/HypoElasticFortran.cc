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


#include <CCA/Components/MPM/ConstitutiveModel/HypoElasticFortran.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h> 
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>

#include <sci_defs/uintah_defs.h>

#include <fstream>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// The following functions are found in fortran/*.F

extern "C"{

#if defined( FORTRAN_UNDERSCORE_END )
#  define HOOKECHK hookechk_
#  define HOOKE_INCREMENTAL hooke_incremental_
#elif defined( FORTRAN_UNDERSCORE_LINUX )
#  define HOOKECHK hookechk_
#  define HOOKE_INCREMENTAL hooke_incremental__
#else // NONE
#  define HOOKECHK hookechk
#  define HOOKE_INCREMENTAL hooke_incremental
#endif

   void HOOKECHK( double UI[], double UJ[], double UK[] );
   void HOOKE_INCREMENTAL( int &nblk, int &ninsv, double &dt, double UI[],
                            double stress[], double D[], double svarg[], double &USM );
}

// End fortran functions.
////////////////////////////////////////////////////////////////////////////////
  

using std::cerr;
using namespace Uintah;

HypoElasticFortran::HypoElasticFortran( ProblemSpecP& ps,MPMFlags* Mflag ) :
  ConstitutiveModel(Mflag)
{
  ps->require("G",d_initialData.G);
  ps->require("K",d_initialData.K);

  double UI[2];
  UI[0]=d_initialData.K;
  UI[1]=d_initialData.G;
  HOOKECHK(UI,UI,UI);
}

HypoElasticFortran::HypoElasticFortran(const HypoElasticFortran* cm) : ConstitutiveModel(cm)
{
  d_initialData.G = cm->d_initialData.G;
  d_initialData.K = cm->d_initialData.K;
}

HypoElasticFortran::~HypoElasticFortran()
{
}

void
HypoElasticFortran::outputProblemSpec( ProblemSpecP& ps,bool output_cm_tag )
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","hypo_elastic_fortran");
  }

  cm_ps->appendElement("G",d_initialData.G);
  cm_ps->appendElement("K",d_initialData.K);
}

HypoElasticFortran*
HypoElasticFortran::clone()
{
  return scinew HypoElasticFortran(*this);
}

void
HypoElasticFortran::initializeCMData( const Patch* patch,
                                      const MPMMaterial* matl,
                                      DataWarehouse* new_dw )
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void
HypoElasticFortran::addParticleState(std::vector<const VarLabel*>& from,
                                          std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
}

void
HypoElasticFortran::computeStableTimestep( const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw )
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

void
HypoElasticFortran::computeStressTensor( const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw )
{
  for(int p=0;p<patches->size();p++){
    double se = 0.0;
    const Patch* patch = patches->get(p);

    Matrix3 Identity; Identity.Identity();
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> pstress,velGrad;
    ParticleVariable<Matrix3> pstress_new;
    constParticleVariable<double> pmass, pvolume, ptemperature;
    constParticleVariable<double> pvolume_new;
    constParticleVariable<Vector> pvelocity;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);

    ParticleVariable<double> pdTdt,p_q;

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel_preReloc,     pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,       pset);
    new_dw->get(pvolume_new,                lb->pVolumeLabel_preReloc,   pset);
    new_dw->get(velGrad,                    lb->pVelGradLabel_preReloc,  pset);

    double UI[2];
    UI[0] = d_initialData.K;
    UI[1] = d_initialData.G;

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*.5;

      // Compute the local sound speed
      double rho_cur = pmass[idx]/pvolume_new[idx];
       
      // This is the (updated) Cauchy stress
      double sigarg[6];
      sigarg[0]=pstress[idx](0,0);
      sigarg[1]=pstress[idx](1,1);
      sigarg[2]=pstress[idx](2,2);
      sigarg[3]=pstress[idx](0,1);
      sigarg[4]=pstress[idx](1,2);
      sigarg[5]=pstress[idx](2,0);
      double Darray[6];
      Darray[0]=D(0,0);
      Darray[1]=D(1,1);
      Darray[2]=D(2,2);
      Darray[3]=D(0,1);
      Darray[4]=D(1,2);
      Darray[5]=D(2,0);
      double svarg[1];
      double USM=9e99;
      double dt = delT;
      int nblk = 1;
      int ninsv = 1;
      HOOKE_INCREMENTAL(nblk, ninsv, dt, UI, sigarg, Darray, svarg, USM);

      pstress_new[idx](0,0) = sigarg[0];
      pstress_new[idx](1,1) = sigarg[1];
      pstress_new[idx](2,2) = sigarg[2];
      pstress_new[idx](0,1) = sigarg[3];
      pstress_new[idx](1,0) = sigarg[3];
      pstress_new[idx](2,1) = sigarg[4];
      pstress_new[idx](1,2) = sigarg[4];
      pstress_new[idx](2,0) = sigarg[5];
      pstress_new[idx](0,2) = sigarg[5];

#if 0
      cout << pstress_new[idx] << endl;
#endif

      c_dil = sqrt(USM/rho_cur);

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume_new[idx]*delT;

      se += e;

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(UI[0]/rho_cur);
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
      new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    }
  }
}

void
HypoElasticFortran::carryForward( const PatchSubset* patches,
                                  const MPMMaterial* matl,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw )
{
  for(int p=0; p<patches->size(); p++) {
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

void
HypoElasticFortran::addComputesAndRequires( Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);
}

void
HypoElasticFortran::addComputesAndRequires( Task*,
                                            const MPMMaterial*,
                                            const PatchSet*,
                                            const bool ) const
{
}

double
HypoElasticFortran::computeRhoMicroCM( double pressure,
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
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR HypoElasticFortran"
       << endl;
#endif
}

void
HypoElasticFortran::computePressEOSCM( double rho_cur, double& pressure,
                                       double p_ref,
                                       double& dp_drho,      double& tmp,
                                       const MPMMaterial* matl, 
                                       double temperature )
{
  //double G = d_initialData.G;
  double bulk = d_initialData.K;
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared

#if 0
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR HypoElasticFortran"
       << endl;
#endif
}

double
HypoElasticFortran::getCompressibility()
{
  return 1.0/d_initialData.K;
}

