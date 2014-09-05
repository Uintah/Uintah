#include "RigidMaterial.h"
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h> 
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

RigidMaterial::RigidMaterial(ProblemSpecP& ps, MPMLabel* Mlb, MPMFlags* Mflag)
{
  lb = Mlb;
  flag = Mflag;
  NGN = 1;
  ps->require("shear_modulus",d_initialData.G);
  ps->require("bulk_modulus",d_initialData.K);
}

RigidMaterial::RigidMaterial(const RigidMaterial* cm)
{
  lb = cm->lb;
  flag = cm->flag;
  NGN = cm->NGN;
  d_initialData.G = cm->d_initialData.G;
  d_initialData.K = cm->d_initialData.K;
}

RigidMaterial::~RigidMaterial()
{
}

void 
RigidMaterial::initializeCMData(const Patch* patch,
                                const MPMMaterial* matl,
                                DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);
  computeStableTimestep(patch, matl, new_dw);
}

void 
RigidMaterial::computeStableTimestep(const Patch* patch,
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  if (pset->numParticles() == 0) return;

  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  // Compute wave speed at the first particle
  particleIndex idx = *(pset->begin());
  double G = d_initialData.G;
  double K = d_initialData.K;
  double c_dil = sqrt((K + 4.*G/3.)*pvolume[idx]/pmass[idx]);
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                   Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                   Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
              lb->delTLabel);
}

void 
RigidMaterial::addComputesAndRequires(Task* task,
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
RigidMaterial::computeStressTensor(const PatchSubset* patches,
                                   const MPMMaterial* matl,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  carryForward(patches, matl, old_dw, new_dw);
}

void 
RigidMaterial::carryForward(const PatchSubset* patches,
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
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(1.e10)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void 
RigidMaterial::addParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to)
{
  // Add the particle state data common to all constitutive models.
  // This method is defined in the ConstitutiveModel base class.
  addSharedParticleState(from, to);
}

double 
RigidMaterial::computeRhoMicroCM(double ,
                                 const double ,
                                 const MPMMaterial* matl)
{
  return matl->getInitialDensity();
}

void 
RigidMaterial::computePressEOSCM(double , double& pressure,
                                 double p_ref,
                                 double& dp_drho, double& tmp,
                                 const MPMMaterial* matl)
{
  double K = d_initialData.K;
  double rho_0 = matl->getInitialDensity();
  pressure = p_ref;
  dp_drho  = K/rho_0;
  tmp = dp_drho;  // speed of sound squared
}

double RigidMaterial::getCompressibility()
{
  return (1.0/d_initialData.K);
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

