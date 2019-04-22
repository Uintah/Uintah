/*
 * 2ProcessNiAlDiffusion.cc
 *
 *  Created on: Jan 25, 2019
 *      Author: jbhooper
 */

#include <CCA/Components/MPM/Diffusion/DiffusionModels/EAM_AlNi_Diffusion.h>
#include <CCA/Components/MPM/Diffusion/DiffusionModels/2ProcessNiAlDiffusion.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <vector>

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolatorFactory.h>
#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>

using namespace Uintah;

NiAl2Process::NiAl2Process(
                            ProblemSpecP      & probSpec
                          , SimulationStateP  & simState
                          , MPMFlags          * mpmFlag
                          , std::string         diff_type
                          )
                          : ScalarDiffusionModel(probSpec,
                                                 simState,
                                                 mpmFlag,
                                                 diff_type)
{
  probSpec->getWithDefault("scaleMultiplier", m_multiplier, 1.0);
  probSpec->getWithDefault("normalized", f_isConcNormalized, true);
  probSpec->require("D0_Liquid", m_D0Liquid);
  probSpec->getWithDefault("D0_Solid", m_D0Solid, 0.0);

  ProblemSpecP interpPS = probSpec->findBlock("function_interp");
  if (!interpPS) {
    throw ProblemSetupException("Diffusion type " + diff_type +
                                " MUST include a function_interp block.",
                                __FILE__, __LINE__);
  }

  m_phaseInterpolator = FunctionInterpolatorFactory::create(interpPS, simState, mpmFlag);

  m_globalMinNiConc = VarLabel::create("globalMinNiConc",min_vartype::getTypeDescription());
  m_globalMinAlConc = VarLabel::create("globalMinAlConc",min_vartype::getTypeDescription());
  m_pRegionType = VarLabel::create("p.regionTypeEnumIndex",ParticleVariable<int>::getTypeDescription());
  m_pRegionType_preReloc = VarLabel::create("p.regionTypeEnumIndex+", ParticleVariable<int>::getTypeDescription());
}

NiAl2Process::~NiAl2Process()
{
  VarLabel::destroy(m_globalMinNiConc);
  VarLabel::destroy(m_globalMinAlConc);
  VarLabel::destroy(m_pRegionType);
  VarLabel::destroy(m_pRegionType_preReloc);
}

void NiAl2Process::addInitialComputesAndRequires(       Task        * task
                                                , const MPMMaterial * matl
                                                , const PatchSet    * patches
                                                ) const
{
  const MaterialSubset  * matlset = matl->thisMaterial();
  task->computes(d_lb->pFluxLabel, matlset);
  // The region type actually requires pConcentration, but because of the way
  //   this initialization is called, it's not listed as a requirement, since
  //   its compute step is flat in the heirarchy with this step.
  task->computes(m_pRegionType, matlset);
  task->computes(m_globalMinNiConc, matlset);
  task->computes(m_globalMinAlConc, matlset);
}

void NiAl2Process::addParticleState(
                                      std::vector<const VarLabel*> &  from
                                   ,  std::vector<const VarLabel*> &  to
                                   ) const {
  from.push_back(d_lb->pFluxLabel);
  from.push_back(m_pRegionType);
  to.push_back(d_lb->pFluxLabel_preReloc);
  to.push_back(m_pRegionType_preReloc);
}

void NiAl2Process::scheduleComputeFlux(       Task        * task
                                      , const MPMMaterial * matl
                                      , const PatchSet    * patches
                                      ) const
{
  const MaterialSubset  * matlSubset  = matl->thisMaterial();
  Ghost::GhostType        gnone       = Ghost::None;

  task->requires(Task::OldDW, d_lb->pConcGradientLabel, matlSubset, gnone);
  task->requires(Task::OldDW, d_lb->pTemperatureLabel,  matlSubset, gnone);
  task->requires(Task::OldDW, m_pRegionType,            matlSubset, gnone);

  task->computes(d_lb->pFluxLabel_preReloc, matlSubset);
  task->computes(d_sharedState->get_delt_label(), getLevel(patches));
}

void NiAl2Process::computeFlux(
                                const Patch         * patch
                              , const MPMMaterial   * matl
                              ,       DataWarehouse * old_dw
                              ,       DataWarehouse * new_dw
                              )
{
  Vector dx     = patch->dCell();
  int    dwIdx  = matl->getDWIndex();

  constParticleVariable<double> pTemperature;
  constParticleVariable<double> pConcentration;
  constParticleVariable<Vector> pGradConcentration;
  constParticleVariable<int>    pRegionType;

  ParticleSubset* pSubset = old_dw->getParticleSubset(dwIdx, patch);

  old_dw->get(pGradConcentration, d_lb->pConcGradientLabel,   pSubset);
  old_dw->get(pTemperature,       d_lb->pTemperatureLabel,    pSubset);
  old_dw->get(pConcentration,     d_lb->pConcentrationLabel,  pSubset);
  old_dw->get(pRegionType,        m_pRegionType,              pSubset);

  ParticleVariable<Vector> pFluxNew;
  new_dw->allocateAndPut(pFluxNew, d_lb->pFluxLabel_preReloc, pSubset);

  size_t numParticles = pSubset->numParticles();
  if (numParticles == 0) return;

  double diffMax = -1e99;
  for (size_t pIdx = 0; pIdx < numParticles; ++pIdx) {
    double minConc;
    EAM_AlNi_Region regionType;
    if (pRegionType[pIdx] == EAM_AlNi_Region::NiRich) {
      regionType = EAM_AlNi_Region::NiRich;
      min_vartype minDiffusantConcentration;
      old_dw->get(minDiffusantConcentration, m_globalMinAlConc);
      minConc = minDiffusantConcentration;
    } else {
      regionType = EAM_AlNi_Region::AlRich;
      min_vartype minDiffusantConcentration;
      old_dw->get(minDiffusantConcentration, m_globalMinNiConc);
      minConc = minDiffusantConcentration;
    }
    const double & Temp = pTemperature[pIdx];
    const double & Conc = pConcentration[pIdx];
    const Vector & gradConc = pGradConcentration[pIdx];
    double D = EAM_AlNi::Diffusivity(Temp,Conc,gradConc,minConc,regionType,
                                     m_phaseInterpolator,m_D0Liquid,m_D0Solid)*m_multiplier;
    pFluxNew[pIdx] = D * pGradConcentration[pIdx];
    diffMax = std::max(diffMax, D);
  }
  // Because we know that the smallest timestep is when diffusivity is
  //   largest...
  double delT_local = computeStableTimeStep(diffMax,dx);
  if (delT_local < 1e-30) {
    std::cerr << "DelT local being poisoned in 2 ProcessNiAlDiffusion.cc  -- Value: " << delT_local << " diffMax: " << diffMax << "\n";
  }
  new_dw->put(delt_vartype(delT_local), d_lb->delTLabel, patch->getLevel());
}

void NiAl2Process::initializeSDMData( const Patch         * patch
                                    , const MPMMaterial   * matl
                                    ,       DataWarehouse * new_dw
                                    ) {
  ParticleVariable<Vector> pFlux;
  ParticleVariable<int>    pRegionType;

  double patchMinConcAl = 1e+10;
  double patchMinConcNi = 1e+10;

  const Vector fluxInitial(0.0);

  int dwIdx = matl->getDWIndex();
  ParticleSubset* pSubset = new_dw->getParticleSubset(dwIdx, patch);

  new_dw->allocateAndPut(pFlux, d_lb->pFluxLabel, pSubset);
  new_dw->allocateAndPut(pRegionType, m_pRegionType, pSubset);

  constParticleVariable<double> pConcentration;
  new_dw->get(pConcentration, d_lb->pConcentrationLabel, pSubset);

  for (ParticleSubset::iterator pIdx = pSubset->begin();
         pIdx < pSubset->end(); ++pIdx) {
    pFlux[*pIdx] = fluxInitial;
    // TODO:  This only works for normalized concentration; should fix for
    //         generalized concentration - 4-2019/JBH
    if (pConcentration[*pIdx] < 0.50) {
      pRegionType[*pIdx] = EAM_AlNi_Region::AlRich;
      patchMinConcNi = std::min(pConcentration[*pIdx],patchMinConcNi);
    }
    else {
      pRegionType[*pIdx] = EAM_AlNi_Region::NiRich;
      // If region is Ni rich and we're tracking normalized concentration of
      //   Ni, then C_Al is 1-C_Ni and track its minimum in Ni rich regions
      //   only.
      patchMinConcAl = std::min(1.0-pConcentration[*pIdx], patchMinConcAl);
    }
  }
  new_dw->put(min_vartype(patchMinConcAl),m_globalMinAlConc);
  new_dw->put(min_vartype(patchMinConcNi),m_globalMinNiConc);

}

void NiAl2Process::addSplitParticlesComputesAndRequires(
                                                               Task        * task
                                                       , const MPMMaterial * matl
                                                       , const PatchSet    * patches
                                                       ) const {

}

void NiAl2Process::splitSDMSpecificParticleData(
                                                 const  Patch   * patch
                                               , const  int       dwIdx
                                               , const  int       nDims
                                               ,        ParticleVariable<int> & pRefOld
                                               ,        ParticleVariable<int> & pRef
                                               , const  unsigned int            oldNumPart
                                               , const  int                     numNewPart
                                               ,        DataWarehouse         * old_dw
                                               ,        DataWarehouse         * new_dw
                                               ) {

}

void NiAl2Process::outputProblemSpec(     ProblemSpecP  & ps
                                    ,     bool            output_rdm_tag
                                    ) const {

}

double NiAl2Process::getMaxConstClamp(const double & Temp) const {
  // Maps the maximum concentration between liquidus and solidus for the Al/Ni rich
  //   regions of the phase diagram to a fit analytic curve.  For systems
  //   where the liquidus of B2 AlNi is not exposed (i.e. other phases are
  //   encountered at 'earlier' concentrations, simulation has shown that
  //   we still approach the liquidus concentration of B2 before crystalizing
  //   below temperatures of about 1100K.  We will ignore multi-phase reactions
  //   below 1100K for now.

  EAM_AlNi_Region regionType = EAM_AlNi_Region::AlRich;
  //   For now we only return things in the Al rich region of the phase diagram
  //     (molePercentNi nominal < 0.50)
  return (EAM_AlNi::getSolidus(Temp, regionType));

}

void NiAl2Process::scheduleTrackConcentrationThreshold(       Task        * task
                                                      ,const  MPMMaterial * matl
                                                      ,const  PatchSet    * patchSet
                                                      ) const {
  {
      Ghost::GhostType gan = Ghost::AroundNodes;

      task->requires(Task::OldDW, m_pRegionType, gan, 0);

      task->requires(Task::NewDW, d_lb->pConcentrationLabel_preReloc, gan, 0);

      task->computes(m_globalMinAlConc);
      task->computes(m_globalMinNiConc);
      task->computes(m_pRegionType_preReloc, matl->thisMaterial());
      //, patchSubset, matlSubset);
  }

}

void NiAl2Process::trackConcentrationThreshold(const Patch          * patch
                                              ,const MPMMaterial    * matl
                                              ,      DataWarehouse  * old_dw
                                              ,      DataWarehouse  * new_dw  ) {

  int    dwIdx  = matl->getDWIndex();

  double localMinAlConc = 1.0e+10;
  double localMinNiConc = 1.0e+10;

  constParticleVariable<int> pRegionType;
  constParticleVariable<double> pConcentrationNew;

  ParticleSubset* pSubset = old_dw->getParticleSubset(dwIdx, patch);
  old_dw->get(pRegionType, m_pRegionType, pSubset);
  new_dw->get(pConcentrationNew, d_lb->pConcentrationLabel_preReloc, pSubset);

  ParticleVariable<int> pRegionType_Update;
  new_dw->allocateAndPut(pRegionType_Update, m_pRegionType_preReloc, pSubset);

  for (size_t pIdx = 0; pIdx < pSubset->numParticles(); ++pIdx) {
    // Set the region type dependent upon the conentration:
    if (pConcentrationNew[pIdx] < 0.50) {
//      pRegionType_Update[pIdx] = EAM_AlNi_Region::AlRich;
    } else {
//      pRegionType_Update[pIdx] = EAM_AlNi_Region::NiRich;
    }
    pRegionType_Update[pIdx] = pRegionType[pIdx];
    if (pRegionType[pIdx] == EAM_AlNi_Region::AlRich) {
      if (pConcentrationNew[pIdx] < localMinNiConc) {
        localMinNiConc = pConcentrationNew[pIdx];
      }
    } else if (pRegionType[pIdx] == EAM_AlNi_Region::AlRich) {
        double C_Al = 1.0-pConcentrationNew[pIdx];
        if (C_Al < localMinAlConc) {
          localMinAlConc = C_Al;
        }
    }
  }
  new_dw->put(min_vartype(localMinAlConc),m_globalMinAlConc);
  new_dw->put(min_vartype(localMinNiConc),m_globalMinNiConc);



}
