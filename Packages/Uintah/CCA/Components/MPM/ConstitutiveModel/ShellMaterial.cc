
#include "ConstitutiveModelFactory.h"
#include "ShellMaterial.h"
#include "MPMMaterial.h"
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h> // just added
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define FRACTURE
#undef FRACTURE

////////////////////////////////////////////////////////////////////////////////
//
// Constructor
//
ShellMaterial::ShellMaterial(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
{
  // Read Material Constants
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  // Initialize labels
  lb = Mlb;

  // Set up support size for interpolation
  d_8or27 = n8or27;
  NGN = 1;
  if (d_8or27 == 27) NGN = 2;
}

////////////////////////////////////////////////////////////////////////////////
//
// Destructor
//
ShellMaterial::~ShellMaterial()
{
}

////////////////////////////////////////////////////////////////////////////////
//
// Make sure all labels are correctly relocated
//
void 
ShellMaterial::addParticleState(std::vector<const VarLabel*>& from,
				std::vector<const VarLabel*>& to)
{
  from.push_back(lb->pInitialThickTopLabel);
  from.push_back(lb->pInitialThickBotLabel);
  from.push_back(lb->pInitialNormalLabel);
  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);

  to.push_back(lb->pInitialThickTopLabel_preReloc);
  to.push_back(lb->pInitialThickBotLabel_preReloc);
  to.push_back(lb->pInitialNormalLabel_preReloc);
  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);
}

////////////////////////////////////////////////////////////////////////////////
//
// Create initialization task graph for local variables
//
void 
ShellMaterial::addInitialComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet*) const
{
}

////////////////////////////////////////////////////////////////////////////////
//
// Initialize the data needed for the Shell Material Model
//
void 
ShellMaterial::initializeCMData(const Patch* patch,
				const MPMMaterial* matl,
				DataWarehouse* new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 One, Zero(0.0); One.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<Matrix3> pDefGrad, pStress;
  new_dw->allocateAndPut(pDefGrad, lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,  lb->pStressLabel,             pset);

  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++) {
    pDefGrad[*iter] = One;
    pStress[*iter] = Zero;
  }

  computeStableTimestep(patch, matl, new_dw);
}

////////////////////////////////////////////////////////////////////////////////
//
// Compute a stable time step.
// This is only called for the initial timestep - all other timesteps
// are computed as a side-effect of compute Stress Tensor
//
void 
ShellMaterial::computeStableTimestep(const Patch* patch,
				     const MPMMaterial* matl,
				     DataWarehouse* new_dw)
{
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);

  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;
  new_dw->get(pmass,     lb->pMassLabel, pset);
  new_dw->get(pvolume,   lb->pVolumeLabel, pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double mu = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    c_dil = sqrt((bulk + 4.0*mu/3.0)*pvolume[idx]/pmass[idx]);
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
ShellMaterial::addComputesAndRequires(Task* task,
				      const MPMMaterial* matl,
				      const PatchSet*) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->pXLabel,                  matlset, gnone);
  task->requires(Task::OldDW, lb->pMassLabel,               matlset, gnone);
  if (d_8or27 == 27)
    task->requires(Task::OldDW, lb->pSizeLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pThickTopLabel,           matlset, gnone);
  task->requires(Task::OldDW, lb->pThickBotLabel,           matlset, gnone);
  task->requires(Task::OldDW, lb->pInitialThickTopLabel,    matlset, gnone);
  task->requires(Task::OldDW, lb->pInitialThickBotLabel,    matlset, gnone);
  task->requires(Task::OldDW, lb->pNormalLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pInitialNormalLabel,      matlset, gnone);
  task->requires(Task::OldDW, lb->pNormalRotRateLabel,      matlset, gnone);
  task->requires(Task::OldDW, lb->pVelocityLabel,           matlset, gnone);
  task->requires(Task::OldDW, lb->pRotationLabel,           matlset, gnone);
  task->requires(Task::OldDW, lb->pDefGradTopLabel,         matlset, gnone);
  task->requires(Task::OldDW, lb->pDefGradCenLabel,         matlset, gnone);
  task->requires(Task::OldDW, lb->pDefGradBotLabel,         matlset, gnone);
  task->requires(Task::OldDW, lb->pStressTopLabel,          matlset, gnone);
  task->requires(Task::OldDW, lb->pStressCenLabel,          matlset, gnone);
  task->requires(Task::OldDW, lb->pStressBotLabel,          matlset, gnone);
  task->requires(Task::OldDW, lb->pStressLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel, matlset, gnone);
  task->requires(Task::NewDW, lb->gVelocityLabel,           matlset, gac, NGN);
  task->requires(Task::NewDW, lb->gNormalRotRateLabel,      matlset, gac, NGN);
  task->requires(Task::OldDW, lb->delTLabel);

  task->computes(lb->pInitialThickTopLabel_preReloc,    matlset);
  task->computes(lb->pInitialThickBotLabel_preReloc,    matlset);
  task->computes(lb->pInitialNormalLabel_preReloc,      matlset);
  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);
  // NOTE: The remaining computes have been declared in ShellMPM
}

void 
ShellMaterial::addComputesAndRequires(Task* ,
				      const MPMMaterial* ,
				      const PatchSet* ,
				      const bool ) const
{
}

////////////////////////////////////////////////////////////////////////////////
//
// Compute the stress tensor
//
void 
ShellMaterial::computeStressTensor(const PatchSubset* patches,
				   const MPMMaterial* matl,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw)
{
  // Initialize contants
  double onethird = (1.0/3.0);
  Matrix3 One; One.Identity();
  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
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
    constParticleVariable<double>  pMass, pThickTop, pThickBot, pThickTop0,
                                   pThickBot0; 
    constParticleVariable<Point>   pX; 
    constParticleVariable<Vector>  pSize, pVelocity, pRotRate, pNormal, 
                                   pNormal0; 
    constParticleVariable<Matrix3> pRotation, pStressTop, pStressCen,
                                   pStressBot, pStress, pDefGradTop,
                                   pDefGradCen, pDefGradBot, pDefGrad;
    constNCVariable<Vector>        gVelocity, gRotRate; 
    delt_vartype                   delT; 
    old_dw->get(pMass,       lb->pMassLabel,               pset);
    old_dw->get(pThickTop,   lb->pThickTopLabel,           pset);
    old_dw->get(pThickBot,   lb->pThickBotLabel,           pset);
    old_dw->get(pX,          lb->pXLabel,                  pset);
    if (d_8or27 == 27)
      old_dw->get(pSize,     lb->pSizeLabel,               pset);
    old_dw->get(pVelocity,   lb->pVelocityLabel,           pset);
    old_dw->get(pRotRate,    lb->pNormalRotRateLabel,      pset);
    old_dw->get(pNormal,     lb->pNormalLabel,             pset);
    old_dw->get(pRotation,   lb->pRotationLabel,           pset);
    old_dw->get(pDefGradTop, lb->pDefGradTopLabel,         pset);
    old_dw->get(pDefGradCen, lb->pDefGradCenLabel,         pset);
    old_dw->get(pDefGradBot, lb->pDefGradBotLabel,         pset);
    old_dw->get(pStressTop,  lb->pStressTopLabel,          pset);
    old_dw->get(pStressCen,  lb->pStressCenLabel,          pset);
    old_dw->get(pStressBot,  lb->pStressBotLabel,          pset);
    old_dw->get(pStress,     lb->pStressLabel,             pset);
    old_dw->get(pDefGrad,    lb->pDeformationMeasureLabel, pset);
    old_dw->get(delT,        lb->delTLabel);
    new_dw->get(gVelocity,   lb->gVelocityLabel,      dwi, patch, gac, NGN);
    new_dw->get(gRotRate,    lb->gNormalRotRateLabel, dwi, patch, gac, NGN);

    // Allocate for updated variables in new_dw 
    ParticleVariable<double>  pVolume_new, pThickTop_new, pThickBot_new,
                              pThickTop0_new, pThickBot0_new; 
    ParticleVariable<Vector>  pNormal_new, pNormal0_new;
    ParticleVariable<Matrix3> pRotation_new, pDefGradTop_new,
                              pDefGradBot_new, pDefGradCen_new,
                              pStressTop_new, pStressCen_new,
                              pStressBot_new, pStress_new, pDefGrad_new;
    new_dw->allocateAndPut(pVolume_new,    lb->pVolumeDeformedLabel,      pset);
    new_dw->allocateAndPut(pThickTop_new,  lb->pThickTopLabel_preReloc,   pset);
    new_dw->allocateAndPut(pThickBot_new,  lb->pThickBotLabel_preReloc,   pset);
    new_dw->allocateAndPut(pThickTop0_new, lb->pInitialThickTopLabel_preReloc,
                           pset);
    new_dw->allocateAndPut(pThickBot0_new, lb->pInitialThickBotLabel_preReloc,
                           pset);
    new_dw->allocateAndPut(pNormal_new,    lb->pNormalLabel_preReloc,     pset);
    new_dw->allocateAndPut(pNormal0_new,   lb->pInitialNormalLabel_preReloc,
                           pset);
    new_dw->allocateAndPut(pRotation_new,  lb->pRotationLabel_preReloc,   pset);
    new_dw->allocateAndPut(pDefGradTop_new,lb->pDefGradTopLabel_preReloc, pset);
    new_dw->allocateAndPut(pDefGradCen_new,lb->pDefGradCenLabel_preReloc, pset);
    new_dw->allocateAndPut(pDefGradBot_new,lb->pDefGradBotLabel_preReloc, pset);
    new_dw->allocateAndPut(pStressTop_new, lb->pStressTopLabel_preReloc,  pset);
    new_dw->allocateAndPut(pStressCen_new, lb->pStressCenLabel_preReloc,  pset);
    new_dw->allocateAndPut(pStressBot_new, lb->pStressBotLabel_preReloc,  pset);
    new_dw->allocateAndPut(pStress_new,    lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pDefGrad_new,  lb->pDeformationMeasureLabel_preReloc,
                           pset);

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

      // Calculate the incremental rotation matrix and store
      Matrix3 Rinc = calcIncrementalRotation(pRotRate[idx], pNormal[idx], delT);
      pRotation_new[idx] = Rinc;

      // Update the normal and store
      pNormal_new[idx] = Rinc*pNormal[idx];
      pNormal0_new[idx] = pNormal0[idx];

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
            velGrad(i+1,j+1) += gvel[i] * d_SXoodx;
            rotGrad(i+1,j+1) += grot[i] * d_SXoodx;
          }
	}
      }

      // Project the velocity gradient and rotation gradient on
      // to surface of the shell
      calcInPlaneGradient(pNormal_new[idx], velGrad, rotGrad);

      // Calculate the layer-wise velocity gradient for stress
      // calculations
      double zTop = pThickTop[idx];
      double zBot = pThickBot[idx];
      Matrix3 rn(pRotRate[idx], pNormal[idx]);
      Matrix3 velGradTop = (velGrad + rotGrad*zTop) + rn ;
      Matrix3 velGradCen = velGrad + rn ;
      Matrix3 velGradBot = (velGrad - rotGrad*zBot) + rn ;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient (F_n^np1 = dudx * dt + Identity).
      Matrix3 defGradIncTop = velGradTop*delT + One;
      Matrix3 defGradIncCen = velGradCen*delT + One;
      Matrix3 defGradIncBot = velGradBot*delT + One;

      // Update the deformation gradient tensor to its time n+1 value.
      pDefGradTop_new[idx] = defGradIncTop*pDefGradTop[idx];
      pDefGradCen_new[idx] = defGradIncCen*pDefGradCen[idx];
      pDefGradBot_new[idx] = defGradIncBot*pDefGradBot[idx];

      // Enforce the no normal stress condition (Sig33 = 0)
      // (we call this condition, roughly, plane stress)
      Matrix3 sigTop(0.0), sigBot(0.0), sigCen(0.0);
      computePlaneStressDefGrad(pDefGradTop_new[idx], sigTop);
      computePlaneStressDefGrad(pDefGradCen_new[idx], sigCen);
      computePlaneStressDefGrad(pDefGradBot_new[idx], sigBot);
      pDefGrad_new[idx] = pDefGradCen_new[idx];

      // Calculate the change in thickness
      double zTopInc = 0.5*(pDefGradTop_new[idx](3,3)+
                            pDefGradCen_new[idx](3,3));
      double zBotInc = 0.5*(pDefGradBot_new[idx](3,3)+
                            pDefGradCen_new[idx](3,3));
      pThickTop_new[idx] = zTopInc*pThickTop0[idx];
      pThickBot_new[idx] = zBotInc*pThickBot0[idx];
      pThickTop0_new[idx] = pThickTop0[idx];
      pThickBot0_new[idx] = pThickBot0[idx];

      // Calculate the total rotation matrix
      Matrix3 R = calcTotalRotation(pNormal0[idx], pNormal_new[idx]);

      // Rotate the stress
      pStressTop_new[idx] = R.Transpose()*sigTop*R;
      pStressCen_new[idx] = R.Transpose()*sigCen*R;
      pStressBot_new[idx] = R.Transpose()*sigBot*R;
      pStress_new[idx] = pStressCen_new[idx];

      // Get the volumetric part of the deformation
      double Je = pDefGrad_new[idx].Determinant();
      pVolume_new[idx]=(pMass[idx]/rho_orig)*Je;

      // Compute the strain energy for all the particles
      double U = 0.5*bulk*(0.5*(Je*Je - 1.0) - log(Je));
      Matrix3 be = pDefGrad_new[idx]*pDefGrad_new[idx].Transpose();
      Matrix3 bebar = be/pow(Je, 2.0/3.0);
      double W = 0.5*shear*(bebar.Trace() - 3.0);
      double e = (U + W)*pVolume_new[idx]/Je;
      strainEnergy += e;

      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.*shear/3.)*pVolume_new[idx]/pMass[idx]);
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

void 
ShellMaterial::computeStressTensor(const PatchSubset* ,
			      const MPMMaterial* ,
			      DataWarehouse* ,
			      DataWarehouse* ,
			      Solver* ,
			      const bool )
{
}
	 
////////////////////////////////////////////////////////////////////////////////
//
// Functions needed by MPMICE
//
// The "CM" versions use the pressure-volume relationship of the CNH model
double 
ShellMaterial::computeRhoMicroCM(double pressure, 
				 const double p_ref,
				 const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;

  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));

  return rho_cur;
}

void 
ShellMaterial::computePressEOSCM(double rho_cur,double& pressure, 
				 double p_ref,
				 double& dp_drho, double& tmp,
				 const MPMMaterial* matl)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = bulk/rho_cur;  // speed of sound squared
}

double 
ShellMaterial::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the incremental rotation matrix for a shell particle
// (WARNING: Can be optimised considerably .. add to TODO list)
// r == rate of rotation of the shell normal
// n == shell normal
// delT == time increment
//
Matrix3 
ShellMaterial::calcIncrementalRotation(const Vector& r, 
                                       const Vector& n,
                                       double delT)
{
  // Calculate the rotation angle
  double phi = r.length()*delT;

  // Create vector a = (n x r)/|(n x r)|
  Vector a = Cross(n,r);
  a /= (a.length()+1.0e-100);

  // Return the incremental rotation matrix
  return Matrix3(phi, a);
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the total rotation matrix for a shell particle
// (WARNING: Can be optimised considerably .. add to TODO list)
// n0 == initial shell normal
// n == current shell normal
//
Matrix3 
ShellMaterial::calcTotalRotation(const Vector& n0, 
                                 const Vector& n)
{
  // Calculate the rotation angle
  double phi = acos(Dot(n0, n)/(n0.length()*n.length()));

  // Find the rotation axis
  Vector a = Cross(n,n0);
  a /= (a.length()+1.0e-100);

  // Return the rotation matrix
  return Matrix3(phi, a);
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the rotation matrix given an angle of rotation and the 
// axis of rotation
// (WARNING: Can be optimised considerably .. add to TODO list)
// Uses the derivative of the Rodriguez vector.
//
Matrix3 
ShellMaterial::calcRotationMatrix(double angle, const Vector& axis)
{
  // Create matrix A = [[0 -a3 a2];[a3 0 -a1];[-a2 a1 0]]
  Matrix3 A(     0.0, -axis[2],  axis[1], 
             axis[2],      0.0, -axis[0], 
            -axis[1],  axis[0],      0.0);
  
  // Calculate the dyad aa
  Matrix3 aa(axis,axis);

  // Initialize the identity matrix
  Matrix3 I; I.Identity();

  // Calculate the rotation matrix
  Matrix3 R = (I - aa)*cos(angle) + aa - A*sin(angle);
  return R;
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the in-plane velocity and rotation gradient.
// 
void
ShellMaterial::calcInPlaneGradient(const Vector& n,
                                   Matrix3& velGrad,
                                   Matrix3& rotGrad)
{
  // Initialize the identity matrix
  Matrix3 I; I.Identity();

  // Calculate the dyad nn
  Matrix3 nn(n,n);

  // Calculate the in-plane identity matrix
  Matrix3 Is = I - nn;

  // Calculate the in-plane velocity and rotation gradients
  velGrad = velGrad*Is;
  rotGrad = rotGrad*Is;
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the plane stress deformation gradient corresponding
// to sig33 = 0 (Use an iterative Newton method)
// WARNING : Assume that the shear components of bbar_elastic are not affected
// when sig33 is set to zero.  Can be optimized considerably later.
//
void
ShellMaterial::computePlaneStressDefGrad(Matrix3& F, Matrix3& sig)
{
  // Initialize bulk, shear
  double bulk = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  Matrix3 One; One.Identity();

  // Other variables
  double J = 1.0;
  double p = 0.0;
  Matrix3 b(0.0);
  Matrix3 s(0.0);
  double Jp = 1.0;
  double pp = 0.0;
  Matrix3 Fp(F);
  Matrix3 bp(0.0);
  Matrix3 sp(0.0);
  double sig33p = 0.0; // Cauchy stress
  double Jm = 1.0;
  double pm = 0.0;
  Matrix3 Fm(F);
  Matrix3 bm(0.0);
  Matrix3 sm(0.0);
  double sig33m = 0.0; // Cauchy stress
  double slope = 0;

  // Initial guess for F(3,3), delta
  double delta = 1.0;
  double epsilon = 1.e-14;
  F(3,3) = 1.0/(F(1,1)*F(2,2));
  do {
    // Central value
    J = F.Determinant();
    p = (0.5*bulk)*(J - 1.0/J);
    b = (F*F.Transpose())/pow(J, 2.0/3.0);
    s = (b - One*(b.Trace()/3.0))*(shear/J);
    sig = One*p + s;

    // Left value
    Fp(3,3) = 1.01*F(3,3);
    Jp = Fp.Determinant();
    pp = (0.5*bulk)*(Jp - 1.0/Jp);
    bp = (Fp*Fp.Transpose())/pow(Jp, 2.0/3.0);
    sp = (bp - One*(bp.Trace()/3.0))*(shear/Jp);
    sig33p = pp + sp(3,3);

    // Right value
    Fm(3,3) = 0.99*F(3,3);
    Jm = Fm.Determinant();
    pm = (0.5*bulk)*(Jm - 1.0/Jm);
    bm = (Fm*Fm.Transpose())/pow(Jm, 2.0/3.0);
    sm = (bm - One*(bm.Trace()/3.0))*(shear/Jm);
    sig33m = pm + sm(3,3);

    // Calculate slope and increment F(3,3)
    slope = (Fp(3,3)-Fm(3,3))/(sig33p-sig33m);
    delta = -sig(3,3)*slope;
    F(3,3) += delta;

  } while (fabs(delta) > epsilon);
  sig(3,3) = 0.0;
}
