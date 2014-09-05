
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
         
IsoHardeningPlastic::IsoHardeningPlastic(const IsoHardeningPlastic* cm)
{
  d_CM.K = cm->d_CM.K;
  d_CM.sigma_0 = cm->d_CM.sigma_0;

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
IsoHardeningPlastic::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet*,
                                   bool ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::ParentOldDW, pAlphaLabel, matlset,Ghost::None);
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
                                               const PatchSet* ,
                                               MPMLabel* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pAlphaLabel_preReloc, matlset, Ghost::None);
  //task->requires(Task::OldDW, pAlphaLabel, matlset, Ghost::None);
}

void IsoHardeningPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
                                            ParticleSubset* addset,
                                            map<const VarLabel*, ParticleVariableBase*>* newState,
                                            ParticleSubset* delset,
                                            DataWarehouse* )
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
 
  ParticleVariable<double> pAlpha;
  constParticleVariable<double> o_Alpha;

  new_dw->allocateTemporary(pAlpha,addset);

  new_dw->get(o_Alpha,pAlphaLabel_preReloc,delset);
  //old_dw->get(o_Alpha,pAlphaLabel,delset);

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
IsoHardeningPlastic::computeFlowStress(const PlasticityState* ,
                                       const double& ,
                                       const double& ,
                                       const MPMMaterial* ,
                                       const particleIndex idx)
{
  double flowStress = d_CM.sigma_0 + d_CM.K*pAlpha[idx];
  return flowStress;
}

double 
IsoHardeningPlastic::computeEpdot(const PlasticityState* state,
                                  const double& ,
                                  const double& ,
                                  const MPMMaterial* ,
                                  const particleIndex)
{
  return state->plasticStrainRate;
}

 
void 
IsoHardeningPlastic::computeTangentModulus(const Matrix3& stress,
                                           const PlasticityState* state,
                                           const double& ,
                                           const MPMMaterial* ,
                                           const particleIndex ,
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
  double shear = state->shearModulus;
  double gamma = 1.0/(1+d_CM.K/(3.0*shear));

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

void
IsoHardeningPlastic::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                                 const particleIndex idx,
                                                 Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}

double
IsoHardeningPlastic::evalDerivativeWRTPlasticStrain(const PlasticityState*,
                                                    const particleIndex )
{
  return d_CM.K;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
IsoHardeningPlastic::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
IsoHardeningPlastic::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
IsoHardeningPlastic::evalDerivativeWRTTemperature(const PlasticityState* ,
                                                  const particleIndex )
{
  return 0.0;
}

double
IsoHardeningPlastic::evalDerivativeWRTStrainRate(const PlasticityState* ,
                                                 const particleIndex )
{
  return 0.0;
}

double
IsoHardeningPlastic::evalDerivativeWRTAlpha(const PlasticityState* ,
                                            const particleIndex )
{
  return d_CM.K;
}



