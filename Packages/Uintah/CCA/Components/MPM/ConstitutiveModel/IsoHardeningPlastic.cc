
#include "IsoHardeningPlastic.h"	
#include <Packages/Uintah/Core/Math/FastMatrix.h>	
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <math.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

IsoHardeningPlastic::IsoHardeningPlastic(ProblemSpecP& ps)
{
  ps->require("K",d_CM.K);
  ps->require("sigma_Y",d_CM.sigma_0);

  // Initialize internal variable labels for evolution
  pAlphaLabel = VarLabel::create("p.alpha",
	ParticleVariable<double>::getTypeDescription());
  pAlphaLabel_preReloc = VarLabel::create("p.alpha+",
	ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
			ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
			ParticleVariable<double>::getTypeDescription());
}
	 
IsoHardeningPlastic::~IsoHardeningPlastic()
{
  VarLabel::destroy(pAlphaLabel);
  VarLabel::destroy(pAlphaLabel_preReloc);
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
}
	 
void 
IsoHardeningPlastic::addInitialComputesAndRequires(Task* task,
						   const MPMMaterial* matl,
						   const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pAlphaLabel, matlset);
  task->computes(pPlasticStrainLabel, matlset);
}

void 
IsoHardeningPlastic::addComputesAndRequires(Task* task,
					    const MPMMaterial* matl,
					    const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, pAlphaLabel, matlset,Ghost::None);
  task->computes(pAlphaLabel_preReloc, matlset);
  task->requires(Task::OldDW, pPlasticStrainLabel, matlset,Ghost::None);
  task->computes(pPlasticStrainLabel_preReloc, matlset);
}

void 
IsoHardeningPlastic::addParticleState(std::vector<const VarLabel*>& from,
				      std::vector<const VarLabel*>& to)
{
  from.push_back(pAlphaLabel);
  to.push_back(pAlphaLabel_preReloc);
  from.push_back(pPlasticStrainLabel);
  to.push_back(pPlasticStrainLabel_preReloc);
}

void 
IsoHardeningPlastic::allocateCMDataAddRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patch,
					       MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,pPlasticStrainLabel, Ghost::None);
  task->requires(Task::OldDW,pAlphaLabel, Ghost::None);
}

void IsoHardeningPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
					    ParticleSubset* addset,
					    map<const VarLabel*, ParticleVariableBase*>* newState,
					    ParticleSubset* delset,
					    DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
 
  ParticleVariable<double> plasticStrain,pAlpha;
  constParticleVariable<double> o_plasticStrain,o_Alpha;

  new_dw->allocateTemporary(plasticStrain,addset);
  new_dw->allocateTemporary(pAlpha,addset);

  old_dw->get(o_plasticStrain,pPlasticStrainLabel,delset);
  old_dw->get(o_Alpha,pAlphaLabel,delset);

  ParticleSubset::iterator o,n = addset->begin();
  for(o = delset->begin(); o != delset->end(); o++, n++) {
    plasticStrain[*n] = o_plasticStrain[*o];
    pAlpha[*n] = o_Alpha[*o];
  }

  (*newState)[pPlasticStrainLabel]=plasticStrain.clone();
  (*newState)[pAlphaLabel]=pAlpha.clone();

}



void 
IsoHardeningPlastic::initializeInternalVars(ParticleSubset* pset,
					    DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pAlpha_new, pAlphaLabel, pset);
  new_dw->allocateAndPut(pPlasticStrain_new, pPlasticStrainLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++) {
    pAlpha_new[*iter] = 0.0;
    pPlasticStrain_new[*iter] = 0.0;
  }
}

void 
IsoHardeningPlastic::getInternalVars(ParticleSubset* pset,
				     DataWarehouse* old_dw) 
{
  old_dw->get(pAlpha, pAlphaLabel, pset);
  old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
}

void 
IsoHardeningPlastic::allocateAndPutInternalVars(ParticleSubset* pset,
						DataWarehouse* new_dw) 
{
  new_dw->allocateAndPut(pAlpha_new, pAlphaLabel_preReloc, pset);
  new_dw->allocateAndPut(pPlasticStrain_new, pPlasticStrainLabel_preReloc,pset);
}

void
IsoHardeningPlastic::updateElastic(const particleIndex idx)
{
  pAlpha_new[idx] = pAlpha[idx];
  pPlasticStrain_new[idx] = pPlasticStrain[idx];
}

void
IsoHardeningPlastic::updatePlastic(const particleIndex idx, 
                                   const double& delGamma)
{
  pAlpha_new[idx] = pAlpha[idx] + sqrt(2.0/3.0)*delGamma;
  pPlasticStrain_new[idx] = pPlasticStrain_new[idx];
}

double
IsoHardeningPlastic::getUpdatedPlasticStrain(const particleIndex idx)
{
  return pPlasticStrain_new[idx];
}

double 
IsoHardeningPlastic::computeFlowStress(const Matrix3& rateOfDeformation ,
				       const double& ,
				       const double& delT,
				       const double& ,
				       const MPMMaterial* ,
				       const particleIndex idx)
{
  // Calculate strain rate and incremental strain
  double edot = sqrt(rateOfDeformation.NormSquared()/1.5);
  if (edot < 0.00001) {
    pPlasticStrain_new[idx] = pPlasticStrain[idx];
  } else {
    pPlasticStrain_new[idx] = pPlasticStrain[idx] + edot*delT;
  }

  double flowStress = d_CM.sigma_0 + d_CM.K*pAlpha[idx];
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
  double gamma = 1.0/(1+d_CM.K/(3.0*shear));

  // Form the elastic-plastic tangent modulus
  for (int ii = 0; ii < 3; ++ii) {
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
                             2.0*shear*gamma*nn(ii1,jj1)*nn(kk1,ll+1);
	}  
      }  
    }  
  }  
}

double
IsoHardeningPlastic::evalDerivativeWRTTemperature(double , double,
                                                  const particleIndex )
{
  return 0.0;
}

double
IsoHardeningPlastic::evalDerivativeWRTStrainRate(double , double,
						 const particleIndex )
{
  return 0.0;
}

double
IsoHardeningPlastic::evalDerivativeWRTAlpha(double , double,
					    const particleIndex )
{
  return d_CM.K;
}

double
IsoHardeningPlastic::evalDerivativeWRTPlasticStrain(double , double,
						    const particleIndex )
{
  ostringstream desc;
  desc << "IsoHardeningPlastic::evalDerivativeWRTPlasticStrain not yet "
       << "implemented. " << endl;
  throw ProblemSetupException(desc.str());
  //return 0.0;
}


void
IsoHardeningPlastic::evalDerivativeWRTScalarVars(double epdot,
						 double T,
						 const particleIndex idx,
						 Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(epdot, T, idx);
  derivs[1] = evalDerivativeWRTTemperature(epdot, T, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(epdot, T, idx);
}
