#include "RigidMaterial.h"
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h> 
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
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
  d_initialData.G = 1.0e200;
  ps->get("shear_modulus",d_initialData.G);
  d_initialData.K = 1.0e200;
  ps->get("bulk_modulus",d_initialData.K);
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
  if (flag->d_integrator == MPMFlags::Implicit) 
    initSharedDataForImplicit(patch, matl, new_dw);
  else {
    initSharedDataForExplicit(patch, matl, new_dw);
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(1.0e10)), 
              lb->delTLabel);
  }
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
  if (flag->d_integrator == MPMFlags::Implicit) {
    addSharedCRForImplicit(task, matlset, patches);
  } else {
    addSharedCRForExplicit(task, matlset, patches);
  }
}

void 
RigidMaterial::computeStressTensor(const PatchSubset* patches,
                                   const MPMMaterial* matl,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  if (flag->d_integrator == MPMFlags::Implicit) {
    computeStressTensorImplicit(patches, matl, old_dw, new_dw);
    return;
  }
  carryForward(patches, matl, old_dw, new_dw);
}

void 
RigidMaterial::computeStressTensorImplicit(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  int dwi = matl->getDWIndex();
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
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
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(1.0)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void
RigidMaterial::addComputesAndRequires(Task* task,
                                      const MPMMaterial* matl,
                                      const PatchSet* patches,
                                      const bool recurse) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForImplicit(task, matlset, patches, recurse);
}

void 
RigidMaterial::computeStressTensor(const PatchSubset* patches,
                                   const MPMMaterial* matl,
                                   DataWarehouse* ,
                                   DataWarehouse* new_dw,
#ifdef HAVE_PETSC
                                   MPMPetscSolver* ,
#else
                                   SimpleSolver* ,
#endif
                                   const bool )

{
  int dwi = matl->getDWIndex();
  DataWarehouse* parent_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  for (int pp = 0; pp < patches->size(); pp++) {
    const Patch* patch = patches->get(pp);
    ParticleSubset* pset = parent_dw->getParticleSubset(dwi, patch);
    carryForwardSharedData(pset, parent_dw, new_dw, matl);
  }
}

void 
RigidMaterial::addParticleState(std::vector<const VarLabel*>& ,
                                std::vector<const VarLabel*>& )
{
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

