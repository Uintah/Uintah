
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
IsoHardeningPlastic::allocateCMDataAddRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patch,
					       MPMLabel* lb) const
{
  //const MaterialSubset* matlset = matl->thisMaterial();
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
 
  ParticleVariable<double> pAlpha;
  constParticleVariable<double> o_Alpha;

  new_dw->allocateTemporary(pAlpha,addset);

  old_dw->get(o_Alpha,pAlphaLabel,delset);

  ParticleSubset::iterator o,n = addset->begin();
  for(o = delset->begin(); o != delset->end(); o++, n++) {
    pAlpha[*n] = o_Alpha[*o];
  }

  (*newState)[pAlphaLabel]=pAlpha.clone();

}



void 
IsoHardeningPlastic::initializeInternalVars(ParticleSubset* pset,
					    DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pAlpha_new, pAlphaLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++) {
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
IsoHardeningPlastic::allocateAndPutRigid(ParticleSubset* pset,
                                         DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pAlpha_new, pAlphaLabel_preReloc, pset);
  // Initializing to zero for the sake of RigidMPM's carryForward
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
     pAlpha_new[*iter] = 0.0;
  }
}

void
IsoHardeningPlastic::updateElastic(const particleIndex idx)
{
  pAlpha_new[idx] = pAlpha[idx];
}

void
IsoHardeningPlastic::updatePlastic(const particleIndex idx, 
                                   const double& delGamma)
{
  pAlpha_new[idx] = pAlpha[idx] + sqrt(2.0/3.0)*delGamma;
}

double 
IsoHardeningPlastic::computeFlowStress(const double& ,
				       const double& ,
				       const double& ,
				       const double& ,
				       const double& ,
				       const MPMMaterial* ,
				       const particleIndex idx)
{
  double flowStress = d_CM.sigma_0 + d_CM.K*pAlpha[idx];
  return flowStress;
}

void 
IsoHardeningPlastic::computeTangentModulus(const Matrix3& stress,
					   const double& , 
					   const double& , 
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
IsoHardeningPlastic::evalDerivativeWRTTemperature(double , double, double,
                                                  const particleIndex )
{
  return 0.0;
}

double
IsoHardeningPlastic::evalDerivativeWRTStrainRate(double , double, double,
						 const particleIndex )
{
  return 0.0;
}

double
IsoHardeningPlastic::evalDerivativeWRTAlpha(double , double, double,
					    const particleIndex )
{
  return d_CM.K;
}

double
IsoHardeningPlastic::evalDerivativeWRTPlasticStrain(double , double, double,
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
						 double ep,
						 double T,
						 const particleIndex idx,
						 Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(epdot, ep, T, idx);
  derivs[1] = evalDerivativeWRTTemperature(epdot, ep, T, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(epdot, ep, T, idx);
}
