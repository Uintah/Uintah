/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include "IsoHardeningFlow.h"        
#include <Core/Math/FastMatrix.h>       
#include <Core/Exceptions/ProblemSetupException.h>
#include <cmath>

using namespace std;
using namespace Uintah;

IsoHardeningFlow::IsoHardeningFlow(ProblemSpecP& ps)
{
  ps->require("K",d_CM.K);
  ps->require("sigma_Y",d_CM.sigma_0);

  // Initialize internal variable labels for evolution
  pAlphaLabel = VarLabel::create("p.alpha",
        ParticleVariable<double>::getTypeDescription());
  pAlphaLabel_preReloc = VarLabel::create("p.alpha+",
        ParticleVariable<double>::getTypeDescription());
}
         
IsoHardeningFlow::IsoHardeningFlow(const IsoHardeningFlow* cm)
{
  d_CM.K = cm->d_CM.K;
  d_CM.sigma_0 = cm->d_CM.sigma_0;

  // Initialize internal variable labels for evolution
  pAlphaLabel = VarLabel::create("p.alpha",
        ParticleVariable<double>::getTypeDescription());
  pAlphaLabel_preReloc = VarLabel::create("p.alpha+",
        ParticleVariable<double>::getTypeDescription());
}
         
IsoHardeningFlow::~IsoHardeningFlow()
{
  VarLabel::destroy(pAlphaLabel);
  VarLabel::destroy(pAlphaLabel_preReloc);
}

void IsoHardeningFlow::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP flow_ps = ps->appendChild("flow_model");
  flow_ps->setAttribute("type","isotropic_hardening");
  flow_ps->appendElement("K",d_CM.K);
  flow_ps->appendElement("sigma_Y",d_CM.sigma_0);
}
         
void 
IsoHardeningFlow::addInitialComputesAndRequires(Task* task,
                                                   const MPMMaterial* matl,
                                                   const PatchSet*)
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pAlphaLabel, matlset);
}

void 
IsoHardeningFlow::addComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet*)
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, pAlphaLabel, matlset,Ghost::None);
  task->computes(pAlphaLabel_preReloc, matlset);
}

void 
IsoHardeningFlow::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet*,
                                   bool /*recurse*/,
                                   bool SchedParent)
{
  const MaterialSubset* matlset = matl->thisMaterial();
  if(SchedParent){
    task->requires(Task::ParentOldDW, pAlphaLabel, matlset,Ghost::None);
  }else{
    task->requires(Task::OldDW, pAlphaLabel, matlset,Ghost::None);
  }
}

void 
IsoHardeningFlow::addParticleState(std::vector<const VarLabel*>& from,
                                      std::vector<const VarLabel*>& to)
{
  from.push_back(pAlphaLabel);
  to.push_back(pAlphaLabel_preReloc);
}

void 
IsoHardeningFlow::allocateCMDataAddRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* ,
                                               MPMLabel* )
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pAlphaLabel_preReloc, matlset, Ghost::None);
  //task->requires(Task::OldDW, pAlphaLabel, matlset, Ghost::None);
}

void IsoHardeningFlow::allocateCMDataAdd(DataWarehouse* new_dw,
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
IsoHardeningFlow::initializeInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pAlpha_new, pAlphaLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++) {
    pAlpha_new[*iter] = 0.0;
  }
}

void 
IsoHardeningFlow::getInternalVars(ParticleSubset* pset,
                                     DataWarehouse* old_dw) 
{
  old_dw->get(pAlpha, pAlphaLabel, pset);
}

void 
IsoHardeningFlow::allocateAndPutInternalVars(ParticleSubset* pset,
                                                DataWarehouse* new_dw) 
{
  new_dw->allocateAndPut(pAlpha_new, pAlphaLabel_preReloc, pset);
}

void
IsoHardeningFlow::allocateAndPutRigid(ParticleSubset* pset,
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
IsoHardeningFlow::updateElastic(const particleIndex idx)
{
  pAlpha_new[idx] = pAlpha[idx];
}

void
IsoHardeningFlow::updatePlastic(const particleIndex idx, 
                                   const double& delGamma)
{
  pAlpha_new[idx] = pAlpha[idx] + sqrt(2.0/3.0)*delGamma;
}

double 
IsoHardeningFlow::computeFlowStress(const PlasticityState* ,
                                       const double& ,
                                       const double& ,
                                       const MPMMaterial* ,
                                       const particleIndex idx)
{
  double flowStress = d_CM.sigma_0 + d_CM.K*pAlpha[idx];
  return flowStress;
}

double 
IsoHardeningFlow::computeEpdot(const PlasticityState* state,
                                  const double& ,
                                  const double& ,
                                  const MPMMaterial* ,
                                  const particleIndex)
{
  return state->plasticStrainRate;
}

 
void 
IsoHardeningFlow::computeTangentModulus(const Matrix3& stress,
                                           const PlasticityState* ,
                                           const double& ,
                                           const MPMMaterial* ,
                                           const particleIndex ,
                                           TangentModulusTensor& ,
                                           TangentModulusTensor& )
{
  throw InternalError("Empty Function: IsoHardeningFlow::computeTangentModulus", __FILE__, __LINE__);
}

void
IsoHardeningFlow::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                                 const particleIndex idx,
                                                 Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}

double
IsoHardeningFlow::evalDerivativeWRTPlasticStrain(const PlasticityState*,
                                                    const particleIndex )
{
  return d_CM.K;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
IsoHardeningFlow::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
IsoHardeningFlow::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
IsoHardeningFlow::evalDerivativeWRTTemperature(const PlasticityState* ,
                                                  const particleIndex )
{
  return 0.0;
}

double
IsoHardeningFlow::evalDerivativeWRTStrainRate(const PlasticityState* ,
                                                 const particleIndex )
{
  return 0.0;
}

double
IsoHardeningFlow::evalDerivativeWRTAlpha(const PlasticityState* ,
                                            const particleIndex )
{
  return d_CM.K;
}
