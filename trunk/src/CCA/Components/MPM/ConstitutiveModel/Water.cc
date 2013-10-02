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

#include <CCA/Components/MPM/ConstitutiveModel/Water.h>
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
#include <iostream>

using namespace std;
using namespace Uintah;

Water::Water(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{

  d_useModifiedEOS = false;
  ps->require("bulk_modulus", d_initialData.d_Bulk);
  ps->require("viscosity",    d_initialData.d_Viscosity);
  ps->require("gamma",        d_initialData.d_Gamma);
}

Water::Water(const Water* cm) : ConstitutiveModel(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;
  d_initialData.d_Bulk = cm->d_initialData.d_Bulk;
  d_initialData.d_Viscosity = cm->d_initialData.d_Viscosity;
  d_initialData.d_Gamma = cm->d_initialData.d_Gamma;
}

Water::~Water()
{
}

void Water::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","water");
  }
  
  cm_ps->appendElement("bulk_modulus",d_initialData.d_Bulk);
  cm_ps->appendElement("viscosity",   d_initialData.d_Viscosity);
  cm_ps->appendElement("gamma",       d_initialData.d_Gamma);
}

Water* Water::clone()
{
  return scinew Water(*this);
}

void Water::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void Water::computeStableTimestep(const Patch* patch,
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

  double bulk = d_initialData.d_Bulk;
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;
     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void Water::computeStressTensor(const PatchSubset* patches,
                                const MPMMaterial* matl,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 tensorL,Shear;
    double J, p,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);
    Matrix3 Identity; Identity.Identity();

    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    constParticleVariable<Matrix3> velGrad;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolume;
    constParticleVariable<Vector> pvelocity;
    ParticleVariable<double> pdTdt,p_q;


    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    new_dw->allocateAndPut(pstress,  lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pdTdt,    lb->pdTdtLabel_preReloc,      pset);
    new_dw->allocateAndPut(p_q,      lb->p_qLabel_preReloc,        pset);
    new_dw->get(deformationGradient_new,
                            lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->get(pvolume,             lb->pVolumeLabel_preReloc,    pset);
    new_dw->get(velGrad,             lb->pVelGradLabel_preReloc,   pset);

    double viscosity = d_initialData.d_Viscosity;
    double bulk  = d_initialData.d_Bulk;
    double gamma = d_initialData.d_Gamma;

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      J = deformationGradient_new[idx].Determinant();

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*0.5;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();

      // Get the deformed volume and current density
      double rho_cur = rho_orig/J;

      // Viscous part of the stress
      Shear = DPrime*(2.*viscosity);

      // get the hydrostatic part of the stress
      double jtotheminusgamma = pow(J,-gamma);
      p = bulk*(jtotheminusgamma - 1.0);

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*(-p) + Shear;

      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt((gamma*jtotheminusgamma*bulk)/rho_cur);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
                                                                                
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
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

void Water::addParticleState(std::vector<const VarLabel*>& ,
                                   std::vector<const VarLabel*>& )
{
  // Add the local particle state data for this constitutive model.
}

void Water::carryForward(const PatchSubset* patches,
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

void Water::addComputesAndRequires(Task* task,
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
Water::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}


// The "CM" versions use the pressure-volume relationship of the CNH model
double Water::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.d_Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;
 
  double p_g_over_bulk = p_gauge/bulk;
  rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));

  return rho_cur;
}

void Water::computePressEOSCM(const double rho_cur,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& tmp,
                                    const MPMMaterial* matl,
                                    double temperature)
{
  double bulk = d_initialData.d_Bulk;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure   = p_ref + p_g;
  dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp        = bulk/rho_cur;  // speed of sound squared
}

double Water::getCompressibility()
{
  return 1.0/d_initialData.d_Bulk;
}


namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(Water::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(Water::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "Water::StateData", 
                                  true, &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
