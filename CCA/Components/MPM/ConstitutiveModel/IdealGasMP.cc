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

#include <CCA/Components/MPM/ConstitutiveModel/IdealGasMP.h>
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
#include <Core/Math/Short27.h> //for Fracture
#include <Core/Grid/Variables/NodeIterator.h> 
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;

IdealGasMP::IdealGasMP(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  ps->require("gamma", d_initialData.gamma);
  ps->require("specific_heat",d_initialData.cv);
  ps->getWithDefault("Pref",d_initialData.Pref,101325.);
}

IdealGasMP::IdealGasMP(const IdealGasMP* cm) : ConstitutiveModel(cm)
{
  d_initialData.gamma = cm->d_initialData.gamma;
  d_initialData.cv = cm->d_initialData.cv;
  d_initialData.Pref = cm->d_initialData.Pref;
}

IdealGasMP::~IdealGasMP()
{
}

void IdealGasMP::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","ideal_gas");
  }

  cm_ps->appendElement("gamma", d_initialData.gamma);
  cm_ps->appendElement("specific_heat",d_initialData.cv);
  cm_ps->appendElement("Pref",d_initialData.Pref);
}


IdealGasMP* IdealGasMP::clone()
{
  return scinew IdealGasMP(*this);
}

void IdealGasMP::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void IdealGasMP::allocateCMDataAddRequires(Task* task,
                                           const MPMMaterial* matl ,
                                           const PatchSet* patches,
                                           MPMLabel* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);
}

void IdealGasMP::allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
    map<const VarLabel*, ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* )
{
  // Copy the data common to all constitutive models from the particle to be 
  // deleted to the particle to be added. 
  // This method is defined in the ConstitutiveModel base class.
  copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);
  
  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
}


void IdealGasMP::addParticleState(std::vector<const VarLabel*>& ,
                                   std::vector<const VarLabel*>& )
{
  // Add the local particle state data for this constitutive model.
}

void IdealGasMP::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume, ptemp;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,        pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,      pset);
  new_dw->get(ptemp,     lb->pTemperatureLabel, pset);
  new_dw->get(pvelocity, lb->pVelocityLabel,    pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double gamma = d_initialData.gamma;
  double cv    = d_initialData.cv;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     double rhoM = pmass[idx]/pvolume[idx];
     double dp_drho = (gamma - 1.0)*cv*ptemp[idx];
     double dp_de   = (gamma - 1.0)*rhoM;

     double p = (gamma - 1.0)*rhoM*cv*ptemp[idx];

     double tmp = dp_drho + dp_de * p /(rhoM * rhoM);

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

void IdealGasMP::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 deformationGradientInc;
    double p,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity;

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass,ptemp;
    ParticleVariable<double> pvolume;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Matrix3> psize;
    constNCVariable<Vector> gvelocity;
    ParticleVariable<double> pdTdt,p_q;
    ParticleVariable<Matrix3>      velGrad;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Ghost::GhostType  gac   = Ghost::AroundCells;

    new_dw->get(gvelocity, lb->gVelocityStarLabel, dwi,patch, gac, NGN);

    old_dw->get(px,                          lb->pXLabel,                 pset);
    old_dw->get(pmass,                       lb->pMassLabel,              pset);
    old_dw->get(ptemp,                       lb->pTemperatureLabel,       pset);
    old_dw->get(psize,                       lb->pSizeLabel,              pset);
    old_dw->get(pvelocity,                   lb->pVelocityLabel,          pset);
    old_dw->get(deformationGradient,         lb->pDeformationMeasureLabel,pset);
    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume,          lb->pVolumeLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,     pset);
    new_dw->allocateAndPut(p_q,              lb->p_qLabel_preReloc,       pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                   lb->pDeformationMeasureLabel_preReloc, pset);
    // Temporary Allocations
    new_dw->allocateTemporary(velGrad,                                    pset);

    double gamma = d_initialData.gamma;
    double cv    = d_initialData.cv;
    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
       particleIndex idx = *iter;

      Matrix3 velGrad_new(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        computeVelocityGradient(velGrad_new,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                                  psize[idx],deformationGradient[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad_new,ni,d_S,S,oodx,gvelocity,px[idx]);
      }

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad_new * delT + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
                                     deformationGradient[idx];

      velGrad[idx] = velGrad_new;
    }

    // The following is used only for pressure stabilization
    CCVariable<double> J_CC;
    new_dw->allocateTemporary(J_CC,       patch);
    J_CC.initialize(0.);
    if(flag->d_doPressureStabilization) {
      CCVariable<double> vol_0_CC;
      CCVariable<double> vol_CC;
      new_dw->allocateTemporary(vol_0_CC, patch);
      new_dw->allocateTemporary(vol_CC,   patch);

      vol_0_CC.initialize(0.);
      vol_CC.initialize(0.);
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        // Get the deformed volume
        pvolume[idx]=(pmass[idx]/rho_orig)*J;

        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        vol_CC[cell_index]  +=pvolume[idx];
        vol_0_CC[cell_index]+=pmass[idx]/rho_orig;
      }

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        J_CC[c]=vol_CC[c]/vol_0_CC[c];
      }
    } //end of pressureStabilization loop  at the patch level

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
       particleIndex idx = *iter;

      // More Pressure Stabilization
      if(flag->d_doPressureStabilization) {
        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        // get the original volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        // Change F such that the determinant is equal to the average for
        // the cell
        deformationGradient_new[idx]*=cbrt(J_CC[cell_index])/cbrt(J);
      }

      double Jold = deformationGradient[idx].Determinant();
      double Jnew = deformationGradient_new[idx].Determinant();
      double Jinc = Jnew/Jold;
      double rhoM = rho_orig/Jnew;
      pvolume[idx]=pmass[idx]/rhoM;
      double dp_drho = (gamma - 1.0)*cv*ptemp[idx];
      double dp_de   = (gamma - 1.0)*rhoM;

      p = (gamma - 1.0)*rhoM*cv*ptemp[idx];

      // try artificial viscosity
      p_q[idx] = 0.;
      if (flag->d_artificial_viscosity) {
        //cerr << "Use the MPM Flag for artificial viscosity" << endl;
        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
        double DTrace = D.Trace();
        if(DTrace<0.){
          double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
          p_q[idx] = 2.5*2.5*dx_ave*dx_ave*rhoM*DTrace*DTrace;
        } else {
          p_q[idx] = 0.;
        }
      }

      // Compute artificial viscosity term
#if 0 // Why is this commented out?
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        //double c_bulk = sqrt(bulk/rho_cur);
        double c_bulk = sqrt(dp_drho);
        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rhoM, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
#endif

      double P = p - d_initialData.Pref;

      double tmp = dp_drho + dp_de * p /(rhoM * rhoM);

      pstress[idx] = Identity*(-P);

      // Temp increase due to P*dV work
      pdTdt[idx] = (-p)*(Jinc-1.)*(1./(rhoM*cv))/delT;

      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt(tmp);
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

    delete interpolator;
  }
}

         
void IdealGasMP::addComputesAndRequires(Task* task,
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
IdealGasMP::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double IdealGasMP::computeRhoMicroCM(double press, 
                                      const double,
                                      const MPMMaterial*,
                                      double Temp,
                                      double rho_guess)
{
  double gamma = d_initialData.gamma;
  double cv    = d_initialData.cv;
  return  press/((gamma - 1.0) * cv * Temp);
}

void IdealGasMP::computePressEOSCM(double rhoM,
                                   double& pressure, 
                                   double,
                                   double& dp_drho, 
                                   double& tmp,
                                   const MPMMaterial*, 
                                   double Temp)
{
  double gamma = d_initialData.gamma;
  double cv    = d_initialData.cv;

  pressure   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  double dp_de   = (gamma - 1.0)*rhoM;
  tmp = dp_drho + dp_de * pressure/(rhoM*rhoM);    // C^2
}

double IdealGasMP::getCompressibility()
{
  return 1.0/d_initialData.Pref;
}

namespace Uintah {
} // End namespace Uintah
