
#include "IsoHardeningPlastic.h"	
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

IsoHardeningPlastic::IsoHardeningPlastic(ProblemSpecP& ps)
{
  ps->require("K",d_const.K);
  ps->require("sigma_Y",d_const.sigma_Y);

  // Initialize internal variable labels for evolution
  pAlphaLabel = VarLabel::create("p.alpha",
				 ParticleVariable<double>::getTypeDescription());
  pAlphaLabel_preReloc = VarLabel::create("p.alpha+",
					  ParticleVariable<double>::getTypeDescription());
}
	 
IsoHardeningPlastic::~IsoHardeningPlastic()
{
  VarLabel::destroy(pAlphaLabel);
  VarLabel::destroy(pAlphaLabel_preReloc);
}
	 
void 
IsoHardeningPlastic::addInitialComputesAndRequires(Task* task,
						   const MPMMaterial* matl,
						   const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pAlphaLabel, matlset);
}

void 
IsoHardeningPlastic::addComputesAndRequires(Task* task,
					    const MPMMaterial* matl,
					    const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, pAlphaLabel, matlset,Ghost::None);
  task->computes(pAlphaLabel_preReloc, matlset);
}

void 
IsoHardeningPlastic::addParticleState(std::vector<const VarLabel*>& from,
				      std::vector<const VarLabel*>& to)
{
  from.push_back(pAlphaLabel);
  to.push_back(pAlphaLabel_preReloc);
}

void 
IsoHardeningPlastic::initializeInternalVars(ParticleSubset* pset,
					    DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pAlpha_new, pAlphaLabel, pset);
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
    pAlpha_new[*iter] = 0.0;
  }
}

void 
IsoHardeningPlastic::getInternalVars(ParticleSubset* pset,
				     DataWarehouse* old_dw) 
{
  old_dw->get(pAlpha, pAlphaLabel, pset);
}

void 
IsoHardeningPlastic::allocateAndPutInternalVars(ParticleSubset* pset,
						DataWarehouse* new_dw) 
{
  new_dw->allocateAndPut(pAlpha_new, pAlphaLabel_preReloc, pset);
}

void
IsoHardeningPlastic::updateElastic(const particleIndex idx)
{
  pAlpha_new[idx] = pAlpha[idx];
}

void
IsoHardeningPlastic::updatePlastic(const particleIndex idx, const double& delGamma)
{
  pAlpha_new[idx] = pAlpha[idx] + sqrt(2.0/3.0)*delGamma;
}

double 
IsoHardeningPlastic::computeFlowStress(const Matrix3& ,
				       const Matrix3& ,
				       const double& ,
				       const double& ,
				       const double& ,
				       const MPMMaterial* ,
				       const particleIndex idx)
{
  double flowStress = d_const.sigma_Y + d_const.K*pAlpha[idx];
  return flowStress;
}

