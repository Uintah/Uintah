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

#include <CCA/Components/MPM/Crack/FractureDefine.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/GUVMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h> // for Fracture
#include <Core/Grid/Variables/NodeIterator.h> // just added
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;

static DebugStream debug_doing("GUVMat_doing", false);
static DebugStream debug("GUVMat", false);
static DebugStream debug_data("GUVMat_data", false);
static DebugStream debug_extra("GUVMat_extra", false);

////////////////////////////////////////////////////////////////////////////////
//
// Constructor
//
GUVMaterial::GUVMaterial(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27):
  ShellMaterial(ps, Mlb, n8or27)
{
  // Read Material Constants
  d_cm.Bulk_lipid = d_initialData.Bulk;
  d_cm.Shear_lipid = d_initialData.Shear;
  //ps->require("bulk_modulus_lipid", d_cm.Bulk_lipid);
  //ps->require("shear_modulus_lipid",d_cm.Shear_lipid);
  ps->require("bulk_modulus_cholesterol", d_cm.Bulk_cholesterol);
  ps->require("shear_modulus_cholesterol",d_cm.Shear_cholesterol);
  debug_doing << "GUVMaterial::GUVMaterial:: Klipid = " << d_cm.Bulk_lipid
        << " Kchol = " << d_cm.Bulk_cholesterol << endl;
}

GUVMaterial::GUVMaterial(const GUVMaterial* cm):ShellMaterial(cm)
{
  d_cm.Bulk_lipid = cm->d_cm.Bulk_lipid;
  d_cm.Shear_lipid = cm->d_cm.Shear_lipid;
  d_cm.Bulk_cholesterol = cm->d_cm.Bulk_cholesterol;
  d_cm.Shear_cholesterol = cm->d_cm.Shear_cholesterol;
}

////////////////////////////////////////////////////////////////////////////////
//
// Destructor
//
GUVMaterial::~GUVMaterial()
{
}

////////////////////////////////////////////////////////////////////////////////
//
// Make sure all labels are correctly relocated
//
void 
GUVMaterial::addParticleState(std::vector<const VarLabel*>& from,
                              std::vector<const VarLabel*>& to)
{
  debug_doing << "GUVMaterial:: Adding particle state." << endl;
  from.push_back(lb->pTypeLabel);
  from.push_back(lb->pThickTopLabel);
  from.push_back(lb->pInitialThickTopLabel);
  from.push_back(lb->pNormalLabel);
  from.push_back(lb->pInitialNormalLabel);

  to.push_back(lb->pTypeLabel_preReloc);
  to.push_back(lb->pThickTopLabel_preReloc);
  to.push_back(lb->pInitialThickTopLabel_preReloc);
  to.push_back(lb->pNormalLabel_preReloc);
  to.push_back(lb->pInitialNormalLabel_preReloc);

  from.push_back(pNormalRotRateLabel); 
  to.push_back(pNormalRotRateLabel_preReloc); 

  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);

  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);
}

////////////////////////////////////////////////////////////////////////////////
//
// Create initialization task graph for local variables
//
void 
GUVMaterial::addInitialComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet*) const
{
  debug_doing << "GUVMaterial:: Adding Initial Computes and Requires." << endl;
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(lb->pTypeLabel,            matlset);
  task->computes(lb->pThickTopLabel,        matlset);
  task->computes(lb->pInitialThickTopLabel, matlset);
  task->computes(lb->pNormalLabel,          matlset);
  task->computes(lb->pInitialNormalLabel,   matlset);

  task->computes(pNormalRotRateLabel, matlset);
}

////////////////////////////////////////////////////////////////////////////////
//
// Initialize the data needed for the GUV Material Model
//
void 
GUVMaterial::initializeCMData(const Patch* patch,
                              const MPMMaterial* matl,
                              DataWarehouse* new_dw)
{
  debug_doing << "GUVMaterial:: Initializing CM Data." << endl;

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 One, Zero(0.0); One.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Vector>  pRotRate; 
  new_dw->allocateAndPut(pRotRate,    pNormalRotRateLabel, pset);
  ParticleVariable<Matrix3> pDefGrad, pStress;
  new_dw->allocateAndPut(pDefGrad, lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,  lb->pStressLabel,             pset);

  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++) {
    particleIndex pidx = *iter;
    pRotRate[pidx] = Vector(0.0,0.0,0.0);
    pDefGrad[pidx] = One;
    pStress[pidx] = Zero;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void 
GUVMaterial::allocateCMDataAddRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patch,
                                       MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->requires(Task::OldDW,pNormalRotRateLabel,         matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pTypeLabel,              matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pThickTopLabel,          matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pInitialThickTopLabel,   matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pNormalLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pInitialNormalLabel,     matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pStressLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel,matlset,Ghost::None);

}


void 
GUVMaterial::allocateCMDataAdd(DataWarehouse* new_dw,
                               ParticleSubset* addset,
                               map<const VarLabel*, ParticleVariableBase*>* newState,
                               ParticleSubset* delset,
                               DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Zero(0.0); 

  ParticleVariable<int>     pType; 
  ParticleVariable<double>  pThick0, pThick; 
  ParticleVariable<Vector>  pNormal0, pNormal, pRotRate; 
  ParticleVariable<Matrix3> pDefGrad, pStress;

  constParticleVariable<int>     o_Type; 
  constParticleVariable<double>  o_Thick0, o_Thick; 
  constParticleVariable<Vector>  o_Normal0, o_Normal, o_RotRate;
  constParticleVariable<Matrix3> o_DefGrad, o_Stress;

  new_dw->allocateTemporary(pType,    addset);
  new_dw->allocateTemporary(pThick0,  addset);
  new_dw->allocateTemporary(pThick,   addset);
  new_dw->allocateTemporary(pNormal0, addset);
  new_dw->allocateTemporary(pNormal,  addset);
  new_dw->allocateTemporary(pRotRate, addset);
  new_dw->allocateTemporary(pDefGrad, addset);
  new_dw->allocateTemporary(pStress,  addset);

  old_dw->get(o_RotRate, pNormalRotRateLabel,         delset);
  old_dw->get(o_Type,    lb->pTypeLabel,              delset);
  old_dw->get(o_Thick0,  lb->pInitialThickTopLabel,   delset);
  old_dw->get(o_Thick,   lb->pThickTopLabel,          delset);
  old_dw->get(o_Normal0, lb->pInitialNormalLabel,     delset);
  old_dw->get(o_Normal,  lb->pNormalLabel,            delset);
  old_dw->get(o_Stress,  lb->pStressLabel,            delset);
  old_dw->get(o_DefGrad, lb->pDeformationMeasureLabel,delset);


  ParticleSubset::iterator o,n = addset->begin();
  for(o=delset->begin(); o != delset->end(); o++,n++) {
    pType[*n]    = o_Type[*o];
    pThick0[*n]  = o_Thick0[*o];
    pThick[*n]   = o_Thick[*o];
    pNormal0[*n] = o_Normal0[*o];
    pNormal[*n]  = o_Normal[*o];
    pRotRate[*n] = o_RotRate[*o];
    pDefGrad[*n] = o_DefGrad[*o];
    pStress[*n]  = o_Stress[*o]; 
  }

  (*newState)[pNormalRotRateLabel]          = pRotRate.clone();
  (*newState)[lb->pTypeLabel]               = pType.clone();
  (*newState)[lb->pInitialThickTopLabel]    = pThick0.clone();
  (*newState)[lb->pThickTopLabel]           = pThick.clone();
  (*newState)[lb->pInitialNormalLabel]      = pNormal0.clone();
  (*newState)[lb->pNormalLabel]             = pNormal.clone();
  (*newState)[lb->pDeformationMeasureLabel] = pDefGrad.clone();
  (*newState)[lb->pStressLabel]             = pStress.clone();

}

////////////////////////////////////////////////////////////////////////////////
//
// Compute a stable time step.
// This is only called for the initial timestep - all other timesteps
// are computed as a side-effect of compute Stress Tensor
//
void 
GUVMaterial::computeStableTimestep(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  debug_doing << "GUVMaterial:: Computing Stable Timestep." << endl;

  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);

  constParticleVariable<int>    pType;
  constParticleVariable<double> pThick, pmass, pvolume;
  constParticleVariable<Vector> pNormal, pvelocity;
  new_dw->get(pType,     lb->pTypeLabel,     pset);
  new_dw->get(pNormal,   lb->pNormalLabel,   pset);
  new_dw->get(pThick,    lb->pThickTopLabel, pset);
  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  debug_data << "GUVMaterial::computeStableTimestep: patch = " << patch
             << " matl = " << matl << " new_dw = " << new_dw << endl;

  double mu_lipid = d_cm.Shear_lipid;
  double K_lipid = d_cm.Bulk_lipid;
  double mu_cholesterol = d_cm.Shear_cholesterol;
  double K_cholesterol = d_cm.Bulk_cholesterol;
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    particleIndex idx = *iter;

    debug_data << "GUVMaterial::computeStableTimestep: particle = " << idx
               << " type = " << pType[idx]
               << " thick = " << pThick[idx]
               << " normal = " << pNormal[idx] << endl;

    // Compute wave speed at each particle, store the maximum
    if (pType[idx] == Lipid)
      c_dil = sqrt((K_lipid + 4.0*mu_lipid/3.0)*pvolume[idx]/pmass[idx]);
    else
      c_dil = sqrt((K_cholesterol + 4.0*mu_cholesterol/3.0)*
                   pvolume[idx]/pmass[idx]);
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  Vector dx = patch->dCell();
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

////////////////////////////////////////////////////////////////////////////////
//
// Create task graph for each time step after initialization
//
void 
GUVMaterial::addComputesAndRequires(Task* task,
                                    const MPMMaterial* matl,
                                    const PatchSet*) const
{
  debug_doing << "GUVMaterial:: Adding Computes and requires." << endl;
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->pXLabel,                  matlset, gnone);
  task->requires(Task::OldDW, lb->pMassLabel,               matlset, gnone);
  if (d_8or27 == 27)
    task->requires(Task::OldDW, lb->pSizeLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pTypeLabel,               matlset, gnone);
  task->requires(Task::OldDW, lb->pThickTopLabel,           matlset, gnone);
  task->requires(Task::OldDW, lb->pInitialThickTopLabel,    matlset, gnone);
  task->requires(Task::OldDW, lb->pNormalLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pVelocityLabel,           matlset, gnone);
  task->requires(Task::OldDW, pNormalRotRateLabel,          matlset, gnone);
  task->requires(Task::OldDW, lb->pStressLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel, matlset, gnone);
  task->requires(Task::NewDW, lb->gVelocityLabel,           matlset, gac, NGN);
  task->requires(Task::NewDW, lb->gNormalRotRateLabel,      matlset, gac, NGN);
  task->requires(Task::OldDW, lb->delTLabel);

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);

  task->computes(lb->pTypeLabel_preReloc,               matlset);
  task->computes(lb->pThickTopLabel_preReloc,           matlset);
  task->computes(lb->pInitialThickTopLabel_preReloc,    matlset);

  task->computes(pAverageMomentLabel,                   matlset);
  task->computes(pNormalDotAvStressLabel,               matlset);
  task->computes(pRotMassLabel,                         matlset);
}


////////////////////////////////////////////////////////////////////////////////
//
// Compute the stress tensor
//
void 
GUVMaterial::computeStressTensor(const PatchSubset* patches,
                                 const MPMMaterial* matl,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  debug_doing << "GUVMaterial:: computing stress tensor." << endl;

  // Initialize contants
  Matrix3 One; One.Identity();
  Matrix3 Zero(0.0);
  double mu_lipid = d_cm.Shear_lipid;
  double K_lipid = d_cm.Bulk_lipid;
  double mu_cholesterol = d_cm.Shear_cholesterol;
  double K_cholesterol = d_cm.Bulk_cholesterol;
  double rho_orig = matl->getInitialDensity();
  Ghost::GhostType gac = Ghost::AroundCells;

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){

    // Current patch
    const Patch* patch = patches->get(pp);

    // Read the datawarehouse
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Read from datawarehouses 
    constParticleVariable<int>     pType;
    constParticleVariable<double>  pMass, pThick, pThick0;
    constParticleVariable<Point>   pX; 
    constParticleVariable<Vector>  pSize, pVelocity, pRotRate, pNormal; 
    constParticleVariable<Matrix3> pStress, pDefGrad;
    constNCVariable<Vector>        gVelocity, gRotRate; 
    delt_vartype                   delT; 
    old_dw->get(pType,       lb->pTypeLabel,               pset);
    old_dw->get(pMass,       lb->pMassLabel,               pset);
    old_dw->get(pThick,      lb->pThickTopLabel,           pset);
    old_dw->get(pThick0,     lb->pInitialThickTopLabel,    pset);
    old_dw->get(pX,          lb->pXLabel,                  pset);
    if (d_8or27 == 27)
      old_dw->get(pSize,     lb->pSizeLabel,               pset);
    old_dw->get(pNormal,     lb->pNormalLabel,             pset);
    old_dw->get(pVelocity,   lb->pVelocityLabel,           pset);
    old_dw->get(pStress,     lb->pStressLabel,             pset);
    old_dw->get(pDefGrad,    lb->pDeformationMeasureLabel, pset);
    old_dw->get(pRotRate,    pNormalRotRateLabel,          pset);
    old_dw->get(delT,        lb->delTLabel);
    new_dw->get(gVelocity,   lb->gVelocityLabel,      dwi, patch, gac, NGN);
    new_dw->get(gRotRate,    lb->gNormalRotRateLabel, dwi, patch, gac, NGN);

    // Allocate for updated variables in new_dw 
    ParticleVariable<int>     pType_new;
    ParticleVariable<double>  pVolume_new, pThick_new, pThick0_new; 
    ParticleVariable<Matrix3> pStress_new, pDefGrad_new;
    new_dw->allocateAndPut(pType_new,      lb->pTypeLabel_preReloc,       pset);
    new_dw->allocateAndPut(pVolume_new,    lb->pVolumeDeformedLabel,      pset);
    new_dw->allocateAndPut(pThick_new,     lb->pThickTopLabel_preReloc,   pset);
    new_dw->allocateAndPut(pThick0_new,    lb->pInitialThickTopLabel_preReloc,
                           pset);
    new_dw->allocateAndPut(pStress_new,    lb->pStressLabel_preReloc,     pset);
    new_dw->allocateAndPut(pDefGrad_new,  lb->pDeformationMeasureLabel_preReloc,
                           pset);

    ParticleVariable<double>  pRotMass;
    ParticleVariable<Vector>  pNDotAvSig;
    ParticleVariable<Matrix3> pAvMoment;
    new_dw->allocateAndPut(pAvMoment,  pAverageMomentLabel,     pset);
    new_dw->allocateAndPut(pNDotAvSig, pNormalDotAvStressLabel, pset);
    new_dw->allocateAndPut(pRotMass,   pRotMassLabel,           pset);

    // Initialize contants
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Initialize variables
    double strainEnergy = 0.0;

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Find the surrounding nodes, interpolation functions and derivatives
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      if (d_8or27 == 27)
        patch->findCellAndShapeDerivatives27(pX[idx], ni, d_S, pSize[idx]);
      else
        patch->findCellAndShapeDerivatives(pX[idx], ni, d_S);

      // Calculate the spatial gradient of the velocity and the 
      // normal rotation rate
      Matrix3 velGrad(0.0), rotGrad(0.0);
      for(int k = 0; k < d_8or27; k++) {
        Vector gvel = gVelocity[ni[k]];
        Vector grot = gRotRate[ni[k]];
        for (int j = 0; j<3; j++){
          double d_SXoodx = d_S[k][j] * oodx[j];
          for (int i = 0; i<3; i++) {
            velGrad(i,j) += gvel[i] * d_SXoodx;
            rotGrad(i,j) += grot[i] * d_SXoodx;
          }
        }
      }
      debug_extra << "GUVMaterial::compStress:: Particle = " << idx
            << " velGrad = " << velGrad
            << " rotGrad = " << rotGrad << endl;

      // Project the velocity gradient and rotation gradient on
      // to surface of the shell
      calcInPlaneGradient(pNormal[idx], velGrad, rotGrad);
      debug_extra << "GUVMaterial::compStress:: Particle = " << idx
            << " in-plane velGrad = " << velGrad
            << " in-plane rotGrad = " << rotGrad << endl;

      // Calculate the layer-wise velocity gradient for stress
      // calculations
      Matrix3 rn(pRotRate[idx], pNormal[idx]);
      debug_extra << "GUVMaterial::compStress:: Particle = " << idx
            << " pNormal = " << pNormal[idx] 
            << " pRotRate = " << pRotRate[idx] << endl;

      Matrix3 velGradCen = velGrad + rn ;
      debug_extra << "GUVMaterial::compStress:: Particle = " << idx
            << " r.n = " << rn 
            << " velGradCen = " << velGradCen << endl;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient (F_n^np1 = dudx * dt + Identity).
      Matrix3 defGradIncCen = velGradCen*delT + One;
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " defGradIncCen = " << defGradIncCen << endl;

      // Calculate the top and bottom Deformation gradient increments
      double h = pThick[idx];
      Matrix3 temp = rotGrad*(0.5*h*delT);
      Matrix3 defGradIncTop = defGradIncCen + temp;
      Matrix3 defGradIncBot = defGradIncCen - temp;

      // Update the deformation gradient tensor to its time n+1 value.
      Matrix3 defGradCen_new = defGradIncCen*pDefGrad[idx];
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " defGradCen_new = " << defGradCen_new << endl;

      // Assume that difference in deformation gradient is very small
      // between top and bottom
      Matrix3 defGradTop = defGradIncTop*pDefGrad[idx];
      Matrix3 defGradBot = defGradIncBot*pDefGrad[idx];

      // Rotate the deformation gradient so that the 33 direction
      // is along the direction of the normal
      Matrix3 R; R.Identity();
      calcTotalRotation(Vector(0,0,1), pNormal[idx], R);
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " R = " << R << endl;
      defGradCen_new = R*defGradCen_new*R.Transpose();
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " rotated  defGradCen_new = " << defGradCen_new << endl;
      
      defGradTop = R*defGradTop*R.Transpose();
      defGradBot = R*defGradBot*R.Transpose();

      // Enforce the no normal stress condition (Sig33 = 0)
      // (we call this condition, roughly, plane stress)
      double K, mu;
      if (pType[idx] == Lipid) {
        K = K_lipid; mu = mu_lipid;
      } else {
        K = K_cholesterol; mu = mu_cholesterol;
      }
      Matrix3 sigCen(0.0);
      if (!computePlaneStressAndDefGrad(defGradCen_new, sigCen, K, mu)) {
        cerr << "Normal = " << pNormal[idx] << endl;
        cerr << "R = " << R << endl;
        cerr << "defGradCen = " << defGradCen_new << endl;
        cerr << "SigCen = " << sigCen << endl;
        exit(1);
      }
      if (idx == 1) 
        cout << "GUVMaterial::compStress:: Particle = " << idx
             << " \n F = " << defGradCen_new << " \n sig = " << sigCen << endl;
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " sigCen = " << sigCen << endl;

      Matrix3 sigTop(0.0);
      if (!computePlaneStressAndDefGrad(defGradTop, sigTop, K, mu)) {
        cerr << "Normal = " << pNormal[idx] << endl;
        cerr << "R = " << R << endl;
        cerr << "defGradTop = " << defGradTop << endl;
        cerr << "SigTop = " << sigTop << endl;
        exit(1);
      }

      Matrix3 sigBot(0.0);
      if (!computePlaneStressAndDefGrad(defGradBot, sigBot, K, mu)) {
        cerr << "Normal = " << pNormal[idx] << endl;
        cerr << "R = " << R << endl;
        cerr << "defGradBot = " << defGradBot << endl;
        cerr << "SigBot = " << sigBot << endl;
        exit(1);
      }

      // Rotate back to global co-ordinates
      defGradCen_new = R.Transpose()*defGradCen_new*R;
      sigCen = R.Transpose()*sigCen*R;
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " back-rotated defGradCen_new = " << defGradCen_new
            << " sigCen = " << sigCen << endl;

      sigTop = R.Transpose()*sigTop*R;
      sigBot = R.Transpose()*sigBot*R;

      // Update the deformation gradients
      pDefGrad_new[idx] = defGradCen_new;

      // Get the volumetric part of the deformation
      double Je = pDefGrad_new[idx].Determinant();
      pVolume_new[idx]=(pMass[idx]/rho_orig)*Je;
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " Je = " << Je
            << " mass = " << pMass[idx]
            << " volume = " << pVolume_new[idx] << endl;

      // Calculate the average stress over the thickness of the shell
      pNDotAvSig[idx] = (pNormal[idx]*sigCen)*pVolume_new[idx];
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " normal = " << pNormal[idx]
            << " n.sig = " << pNDotAvSig[idx] << endl;

      // Copy variables
      pType_new[idx] = pType[idx];
      pThick0_new[idx] = pThick0[idx];
      pThick_new[idx] = pThick[idx];
      
      // Update the stresses
      pStress_new[idx] = sigCen;

      // Calculate the average moment over the thickness of the shell
      Matrix3 nn(pNormal[idx], pNormal[idx]);
      Matrix3 Is = One - nn;
      Matrix3 avMoment = (sigTop - sigBot)*(0.5*h);
      pAvMoment[idx] = (Is*avMoment*Is)*pVolume_new[idx];

      // Calculate inertia term
      pRotMass[idx] = pMass[idx]*h*h/12.0;
      debug << "GUVMaterial::compStress:: Particle = " << idx
            << " pAvMoment = " << pAvMoment[idx]
            << " pRotMass = " << pRotMass[idx] << endl;

      // Compute the strain energy for all the particles
      Matrix3 be = pDefGrad_new[idx]*pDefGrad_new[idx].Transpose();
      Matrix3 bebar = be/pow(Je, 2.0/3.0);
      double U = 0.5*K*(0.5*(Je*Je - 1.0) - log(Je));
      double W = 0.5*mu*(bebar.Trace() - 3.0);
      double c_dil = sqrt((K + 4.*mu/3.)*pVolume_new[idx]/pMass[idx]);

      double e = (U + W)*pVolume_new[idx]/Je;
      strainEnergy += e;
      Vector pVel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(sum_vartype(strainEnergy), lb->StrainEnergyLabel);
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Add computes and requires update of rotation rate
//
void 
GUVMaterial::addComputesRequiresRotRateUpdate(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* ) 
{
  debug_doing << "GUVMaterial:: Adding computes/requires for rot rate Update." 
              << endl;
  Ghost::GhostType gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pMassLabel,              matlset, gnone);
  task->requires(Task::OldDW, lb->pNormalLabel,            matlset, gnone);
  task->requires(Task::OldDW, lb->pInitialNormalLabel,     matlset, gnone);
  task->requires(Task::OldDW, pNormalRotRateLabel,         matlset, gnone);
  task->requires(Task::NewDW, lb->pVolumeDeformedLabel,    matlset, gnone);
  task->requires(Task::NewDW, lb->pThickTopLabel_preReloc, matlset, gnone);
  task->requires(Task::NewDW, pNormalRotAccLabel,          matlset, gnone);

  task->computes(lb->pNormalLabel_preReloc,             matlset);
  task->computes(lb->pInitialNormalLabel_preReloc,      matlset);
  task->computes(pNormalRotRateLabel_preReloc,          matlset);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually update rotation rate
//
void 
GUVMaterial::particleNormalRotRateUpdate(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  debug_doing << "GUVMaterial:: computing normal rot rate Update." << endl;

  // Constants
  Matrix3 One; One.Identity();
  double K_lipid   = d_cm.Shear_lipid;
  double mu_lipid  = d_cm.Bulk_lipid;
  double K_cholesterol   = d_cm.Shear_cholesterol;
  double mu_cholesterol  = d_cm.Bulk_cholesterol;
  double E_lipid   = 9.0*K_lipid*mu_lipid/(3.0*K_lipid+mu_lipid);
  double E_cholesterol   = 9.0*K_cholesterol*mu_cholesterol/
    (3.0*K_cholesterol+mu_cholesterol);
  int    dwi = matl->getDWIndex();
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  // Local storage
  constParticleVariable<int>     pType;
  constParticleVariable<double>  pMass, pVol, pThick;
  constParticleVariable<Vector>  pNormal, pNormal0, pRotRate, pRotAcc;
  ParticleVariable<Vector>       pRotRate_new, pNormal_new, pNormal0_new;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get the needed data
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pType,     lb->pTypeLabel,              pset);
    old_dw->get(pMass,     lb->pMassLabel,              pset);
    old_dw->get(pNormal,   lb->pNormalLabel,            pset);
    old_dw->get(pNormal0,  lb->pInitialNormalLabel,     pset);
    old_dw->get(pRotRate,  pNormalRotRateLabel,         pset);
    new_dw->get(pThick,    lb->pThickTopLabel_preReloc, pset);
    new_dw->get(pVol,      lb->pVolumeDeformedLabel,    pset);
    new_dw->get(pRotAcc,   pNormalRotAccLabel,          pset);

    // Allocate the updated particle variables
    new_dw->allocateAndPut(pNormal_new,  lb->pNormalLabel_preReloc,     pset);
    new_dw->allocateAndPut(pNormal0_new, lb->pInitialNormalLabel_preReloc,
                           pset);
    new_dw->allocateAndPut(pRotRate_new, pNormalRotRateLabel_preReloc, pset);

    // Loop over particles
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Calculate the tilde rot rate
      Vector rotRateTilde = pRotAcc[idx]*delT;
      debug << "GUVMaterial::RotRateUpd:: Particle = " << idx
            << " pRotAcc = " << pRotAcc[idx] << " delT = " << delT 
            << " rotRateTilde = " << rotRateTilde << endl;

      // Calculate the in-surface identity tensor
      Matrix3 nn(pNormal[idx], pNormal[idx]);
      Matrix3 Is = One - nn;
      debug << "GUVMaterial::RotRateUpd:: Particle = " << idx
            << " nn = " << nn << " Is = " << Is << endl;

      // The small value of thickness requires the following
      // implicit correction step (** WARNING ** Taken from cfdlib code)
      double hh = pThick[idx];

      double E;
      if (pType[idx] == Lipid) E = E_lipid; 
      else E = E_cholesterol; 
      double fac = 6.0*E*(pVol[idx]/pMass[idx])*pow(delT/hh, 2);
      Is = One + Is*fac; 
      Vector corrRotRateTilde = rotRateTilde;
      //Vector corrRotRateTilde(0.0,0.0,0.0);
      //Is.solveCramer(rotRateTilde, corrRotRateTilde);

      // Update the particle's rotational velocity
      pRotRate_new[idx] = pRotRate[idx] + corrRotRateTilde;

      debug << "GUVMaterial::RotRateUpd:: Particle = " << idx
            << " pRotRate = " << pRotRate[idx] 
            << " corrRotRateTilde = " << corrRotRateTilde
            << " pRotRate_new = " << pRotRate_new[idx] << endl;
           
      // Calculate the incremental rotation matrix and store
      Matrix3 Rinc = calcIncrementalRotation(pRotRate_new[idx], pNormal[idx], 
                                             delT);

      // Update the normal 
      pNormal_new[idx] = Rinc*pNormal[idx];
      double len = pNormal_new[idx].length();
      ASSERT(len > 0.0);
      pNormal_new[idx] = pNormal_new[idx]/len;
      pNormal0_new[idx] = pNormal0[idx];

      // Rotate the rotation rate
      pRotRate_new[idx] = Rinc*pRotRate_new[idx];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//
// Functions needed by MPMICE
//
// The "CM" versions use the pressure-volume relationship of the CNH model
double 
GUVMaterial::computeRhoMicroCM(double pressure, 
                               const double p_ref,
                               const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = 0.5*(d_cm.Bulk_lipid+d_cm.Bulk_cholesterol);

  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));

  return rho_cur;
}

void 
GUVMaterial::computePressEOSCM(double rho_cur,double& pressure, 
                               double p_ref,
                               double& dp_drho, double& tmp,
                               const MPMMaterial* matl)
{
  double bulk = 0.5*(d_cm.Bulk_lipid+d_cm.Bulk_cholesterol);
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = bulk/rho_cur;  // speed of sound squared
}

double 
GUVMaterial::getCompressibility()
{
  double bulk = 0.5*(d_cm.Bulk_lipid+d_cm.Bulk_cholesterol);
  return 1.0/bulk;
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the plane stress deformation gradient corresponding
// to sig33 = 0 (Use an iterative Newton method)
//
bool
GUVMaterial::computePlaneStressAndDefGrad(Matrix3& F, Matrix3& sig, 
                                            double bulk, double shear)
{
  /*
  double epsilon = 1.e-14;
  double delta = 1.;
  double f33, f33p, jv, f33m, jvp, jvm, sig33, sig33p, sig33m;

  // Guess F33
  f33 =  1./(F(0,0)*F(1,1));

  // Find F33 that enforces plane stress
  while(fabs(delta) > epsilon){
    double detF2=(F(0,0)*F(1,1) - F(1,0)*F(0,1));
    jv = f33*detF2;
    double FinF = F(0,0)*F(0,0)+F(0,1)*F(0,1)+F(1,0)*F(1,0)+F(1,1)*F(1,1);
    sig33 = (shear/(3.*pow(jv,2./3.)))*
            (2.*f33*f33 - FinF) + (.5*bulk)*(jv - 1./jv);

    f33p = 1.01*f33;
    f33m = 0.99*f33;
    jvp = f33p*detF2;
    jvm = f33m*detF2;

    sig33p = (shear/(3.*pow(jvp,2./3.)))*
                (2.*f33p*f33p - FinF) + (.5*bulk)*(jvp - 1./jvp);

    sig33m = (shear/(3.*pow(jvm,2./3.)))*
                (2.*f33m*f33m - FinF) + (.5*bulk)*(jvm - 1./jvm);

    delta = -sig33/((sig33p-sig33m)/(f33p-f33m));

    f33 = f33 + delta;
  }

  // Update F
  F(0,2) = 0.0; F(2,0) = 0.0; F(1,2) = 0.0; F(2,1) = 0.0;
  F(2,2) = f33;

  */
  // Calculate Jacobian
  double J = F.Determinant();
  if (!(J > 0.0)) {
    cerr << "GUVMaterial::** ERROR ** F = " << F << " det F = " << J << endl;
    return false;
  }

  // Calcuate Kirchhoff stress
  Matrix3 I; I.Identity();
  double Jp = (0.5*bulk)*(J*J - 1.0);
  Matrix3 b = (F*F.Transpose())*pow(J, -(2.0/3.0));
  Matrix3 Js = (b - I*(b.Trace()/3.0))*shear;
  Matrix3 tau = I*Jp + Js;

  sig = tau/J;
  return true;
}

