
#include "IsoHardeningPlastic.h"	
#include <Packages/Uintah/Core/Math/FastMatrix.h>	
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

void 
IsoHardeningPlastic::computeTangentModulus(const Matrix3& stress,
					   const Matrix3& , 
					   double ,
					   double ,
                                           const particleIndex ,
                                           const MPMMaterial* ,
					   TangentModulusTensor& Ce,
					   TangentModulusTensor& Cep)
{
  // Calculate the deviatoric stress
  Matrix3 one; one.Identity();
  Matrix3 sigdev = stress - one*(stress.Trace()/3.0);

  // Calculate the equivalent stress
  double sigeqv = sqrt(sigdev.NormSquared()); 

  // Calculate direction of plastic flow
  Matrix3 nn = sigdev*(1.0/sigeqv);

  // Calculate gamma
  double shear = Ce(0,1,0,1);
  double gamma = 1.0/(1+d_const.K/(3.0*shear));

  // Form the elastic-plastic tangent modulus
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
	for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
                             2.0*shear*gamma*nn(ii,jj)*nn(kk,ll);
	}  
      }  
    }  
  }  
}
