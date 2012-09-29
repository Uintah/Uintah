/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/Angio/AngioMaterial.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Wasatch/WasatchMaterial.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

SimulationState::SimulationState(ProblemSpecP &ps)
{
   VarLabel* nonconstDelt = 
     VarLabel::create("delT", delt_vartype::getTypeDescription() );
// ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());
   nonconstDelt->allowMultipleComputes();
   delt_label = nonconstDelt;

   refineFlag_label = VarLabel::create("refineFlag",
                                       CCVariable<int>::getTypeDescription());
   oldRefineFlag_label = VarLabel::create("oldRefineFlag",
                                       CCVariable<int>::getTypeDescription());
   refinePatchFlag_label = VarLabel::create("refinePatchFlag",
                                       PerPatch<int>::getTypeDescription());
   switch_label = VarLabel::create("switchFlag", 
                                   max_vartype::getTypeDescription());

   d_elapsed_time = 0.0;
   d_needAddMaterial = 0;

  d_lockstepAMR = false;
  ProblemSpecP amr = ps->findBlock("AMR");
  if (amr)
    amr->get("useLockStep", d_lockstepAMR);

  all_mpm_matls = 0;
  all_cz_matls = 0;
  all_angio_matls = 0;
  all_ice_matls = 0;
  all_wasatch_matls = 0;  
  all_arches_matls = 0;
  all_matls = 0;
  orig_all_matls = 0;
  allInOneMatl = 0;
  max_matl_index = 0;
  refine_flag_matls = 0;
  d_isCopyDataTimestep = 0;
  d_isRegridTimestep = 0;

  d_switchState = false;
  d_simTime = 0;
  d_numDims = 0;
  d_activeDims[0] = d_activeDims[1] = d_activeDims[2] = 0;
  //initialize the overhead percentage
  overheadIndex=0;
  overheadAvg=0;
  for(int i=0;i<OVERHEAD_WINDOW;i++)
  {
    double x=i/(OVERHEAD_WINDOW/2);
    overheadWeights[i]=8-x*x*x;
    overhead[i]=0;
  }

  clearStats();  
}

void SimulationState::registerMaterial(Material* matl)
{
   matl->registerParticleState(this);
   matl->setDWIndex((int)matls.size());

   matls.push_back(matl);
   if ((int)matls.size() > max_matl_index) {
     max_matl_index = matls.size();
   }

   if(matl->hasName())
     named_matls[matl->getName()] = matl;
}

void SimulationState::registerMaterial(Material* matl,unsigned int index)
{
   matl->registerParticleState(this);
   matl->setDWIndex(index);

   if (matls.size() <= index)
     matls.resize(index+1);
   matls[index]=matl;

   if ((int)matls.size() > max_matl_index) {
     max_matl_index = matls.size();
   }

   if(matl->hasName())
     named_matls[matl->getName()] = matl;
}


void SimulationState::registerMPMMaterial(MPMMaterial* matl)
{
  mpm_matls.push_back(matl);
  registerMaterial(matl);
}

void SimulationState::registerMPMMaterial(MPMMaterial* matl,unsigned int index)
{
  mpm_matls.push_back(matl);
  registerMaterial(matl,index);
}

void SimulationState::registerCZMaterial(CZMaterial* matl)
{
  cz_matls.push_back(matl);
  registerMaterial(matl);
}

void SimulationState::registerCZMaterial(CZMaterial* matl,unsigned int index)
{
  cz_matls.push_back(matl);
  registerMaterial(matl,index);
}

void SimulationState::registerAngioMaterial(AngioMaterial* matl)
{
  angio_matls.push_back(matl);
  registerMaterial(matl);
}

void SimulationState::registerAngioMaterial(AngioMaterial* matl,
                                            unsigned int index)
{
  angio_matls.push_back(matl);
  registerMaterial(matl,index);
}

void SimulationState::registerArchesMaterial(ArchesMaterial* matl)
{
   arches_matls.push_back(matl);
   registerMaterial(matl);
}

void SimulationState::registerICEMaterial(ICEMaterial* matl)
{
   ice_matls.push_back(matl);
   registerMaterial(matl);
}

void SimulationState::registerICEMaterial(ICEMaterial* matl,unsigned int index)
{
   ice_matls.push_back(matl);
   registerMaterial(matl,index);
}

void SimulationState::registerWasatchMaterial(WasatchMaterial* matl)
{
  wasatch_matls.push_back(matl);
  registerMaterial(matl);
}

void SimulationState::registerWasatchMaterial(WasatchMaterial* matl,unsigned int index)
{
  wasatch_matls.push_back(matl);
  registerMaterial(matl,index);
}

void SimulationState::registerSimpleMaterial(SimpleMaterial* matl)
{
  simple_matls.push_back(matl);
  registerMaterial(matl);
}

void SimulationState::finalizeMaterials()
{
  if (all_mpm_matls && all_mpm_matls->removeReference())
    delete all_mpm_matls;
  all_mpm_matls = scinew MaterialSet();
  all_mpm_matls->addReference();
  vector<int> tmp_mpm_matls(mpm_matls.size());
  for( int i=0; i<(int)mpm_matls.size(); i++ ) {
    tmp_mpm_matls[i] = mpm_matls[i]->getDWIndex();
  }
  all_mpm_matls->addAll(tmp_mpm_matls);

  if (all_cz_matls && all_cz_matls->removeReference())
    delete all_cz_matls;
  all_cz_matls = scinew MaterialSet();
  all_cz_matls->addReference();
  vector<int> tmp_cz_matls(cz_matls.size());
  for( int i=0; i<(int)cz_matls.size(); i++ ) {
    tmp_cz_matls[i] = cz_matls[i]->getDWIndex();
  }
  all_cz_matls->addAll(tmp_cz_matls);
  
  if (all_angio_matls && all_angio_matls->removeReference())
    delete all_angio_matls;
  all_angio_matls = scinew MaterialSet();
  all_angio_matls->addReference();
  vector<int> tmp_angio_matls(angio_matls.size());
  for( int i=0; i<(int)angio_matls.size(); i++ ) {
    tmp_angio_matls[i] = angio_matls[i]->getDWIndex();
  }
  all_angio_matls->addAll(tmp_angio_matls);
  
  if (all_arches_matls && all_arches_matls->removeReference())
    delete all_arches_matls;
  all_arches_matls = scinew MaterialSet();
  all_arches_matls->addReference();
  vector<int> tmp_arches_matls(arches_matls.size());
  for (int i = 0; i<(int)arches_matls.size();i++)
    tmp_arches_matls[i] = arches_matls[i]->getDWIndex();
  all_arches_matls->addAll(tmp_arches_matls);

  if (all_ice_matls && all_ice_matls->removeReference())
    delete all_ice_matls;
  all_ice_matls = scinew MaterialSet();
  all_ice_matls->addReference();
  vector<int> tmp_ice_matls(ice_matls.size());
  for(int i=0;i<(int)ice_matls.size();i++)
    tmp_ice_matls[i] = ice_matls[i]->getDWIndex();
  all_ice_matls->addAll(tmp_ice_matls);

  if (all_wasatch_matls && all_wasatch_matls->removeReference())
    delete all_wasatch_matls;
  all_wasatch_matls = scinew MaterialSet();
  all_wasatch_matls->addReference();
  vector<int> tmp_wasatch_matls(wasatch_matls.size());
  for(int i=0;i<(int)wasatch_matls.size();i++)
    tmp_wasatch_matls[i] = wasatch_matls[i]->getDWIndex();
  all_wasatch_matls->addAll(tmp_wasatch_matls);
  
  if (all_matls && all_matls->removeReference())
    delete all_matls;
  all_matls = scinew MaterialSet();
  all_matls->addReference();
  vector<int> tmp_matls(matls.size());
  for(int i=0; i<(int)matls.size(); i++) {
    tmp_matls[i] = matls[i]->getDWIndex();
  }
  all_matls->addAll(tmp_matls);

  if (orig_all_matls == 0) {
    orig_all_matls = scinew MaterialSet();
    orig_all_matls->addReference();
    orig_all_matls->addAll(tmp_matls);
  }

  if (allInOneMatl && allInOneMatl->removeReference())
    delete allInOneMatl;
  allInOneMatl = scinew MaterialSubset();
  allInOneMatl->addReference();
  // a material that represents all materials 
  // (i.e. summed over all materials -- the whole enchilada)
  allInOneMatl->add((int)matls.size());

  //refine matl subset, only done on matl 0 (matl independent)
  if (!refine_flag_matls) {
    refine_flag_matls = scinew MaterialSubset();
    refine_flag_matls->addReference();
    refine_flag_matls->add(0);
  }
}

void SimulationState::clearMaterials()
{
  for (int i = 0; i < (int)matls.size(); i++)
    old_matls.push_back(matls[i]);

  if(all_matls && all_matls->removeReference())
    delete all_matls;
  
  if(all_mpm_matls && all_mpm_matls->removeReference())
    delete all_mpm_matls;

  if(all_cz_matls && all_cz_matls->removeReference())
    delete all_cz_matls;

  if(all_angio_matls && all_angio_matls->removeReference())
    delete all_angio_matls;

  if (all_arches_matls && all_arches_matls->removeReference())
    delete all_arches_matls;

  if(all_ice_matls && all_ice_matls->removeReference())
    delete all_ice_matls;

  if(all_wasatch_matls && all_wasatch_matls->removeReference())
    delete all_wasatch_matls;

  if (allInOneMatl && allInOneMatl->removeReference()) {
    delete allInOneMatl;
  }

  matls.clear();
  mpm_matls.clear();
  cz_matls.clear();
  angio_matls.clear();
  arches_matls.clear();
  ice_matls.clear();
  wasatch_matls.clear();
  simple_matls.clear();
  named_matls.clear();
  d_particleState.clear();
  d_particleState_preReloc.clear();
  d_cohesiveZoneState.clear();
  d_cohesiveZoneState_preReloc.clear();

  all_matls         = 0;
  all_mpm_matls     = 0;
  all_cz_matls      = 0;
  all_angio_matls   = 0;
  all_arches_matls  = 0;
  all_ice_matls     = 0;
  all_wasatch_matls = 0;
  allInOneMatl      = 0;
}

SimulationState::~SimulationState()
{
  VarLabel::destroy(delt_label);
  VarLabel::destroy(refineFlag_label);
  VarLabel::destroy(oldRefineFlag_label);
  VarLabel::destroy(refinePatchFlag_label);
  VarLabel::destroy(switch_label);
  clearMaterials();

  for (unsigned i = 0; i < old_matls.size(); i++)
    delete old_matls[i];

  if(refine_flag_matls && refine_flag_matls->removeReference())
    delete refine_flag_matls;

  if(orig_all_matls && orig_all_matls->removeReference())
    delete orig_all_matls;
}

const MaterialSet* SimulationState::allMPMMaterials() const
{
  ASSERT(all_mpm_matls != 0);
  return all_mpm_matls;
}

const MaterialSet* SimulationState::allCZMaterials() const
{
  ASSERT(all_cz_matls != 0);
  return all_cz_matls;
}

const MaterialSet* SimulationState::allAngioMaterials() const
{
  ASSERT(all_angio_matls != 0);
  return all_angio_matls;
}

const MaterialSet* SimulationState::allArchesMaterials() const
{
  ASSERT(all_arches_matls != 0);
  return all_arches_matls;
}

const MaterialSet* SimulationState::allICEMaterials() const
{
  ASSERT(all_ice_matls != 0);
  return all_ice_matls;
}

const MaterialSet* SimulationState::allWasatchMaterials() const
{
  ASSERT(all_wasatch_matls != 0);
  return all_wasatch_matls;
}

const MaterialSet* SimulationState::allMaterials() const
{
  ASSERT(all_matls != 0);
  return all_matls;
}

const MaterialSet* SimulationState::originalAllMaterials() const
{
  ASSERT(orig_all_matls != 0);
  return orig_all_matls;
}

void SimulationState::setOriginalMatlsFromRestart(MaterialSet* matls)
{
  if (orig_all_matls && orig_all_matls->removeReference())
    delete orig_all_matls;
  orig_all_matls = matls;
}
  

const MaterialSubset* SimulationState::refineFlagMaterials() const
{
  ASSERT(refine_flag_matls != 0);
  return refine_flag_matls;
}

Material* SimulationState::getMaterialByName(const std::string& name) const
{
  map<string, Material*>::const_iterator iter = named_matls.find(name);
  if(iter == named_matls.end())
    return 0;
  return iter->second;
}

//__________________________________
//
Material* SimulationState::parseAndLookupMaterial(ProblemSpecP& params,
                                                  const std::string& name) const
{
  // for single material problems return matl 0
  Material* result = getMaterial(0);

  if( getNumMatls() > 1){
    string matlname;
    if(!params->get(name, matlname)){
      throw ProblemSetupException("Cannot find material section", __FILE__, __LINE__);
    }

    result = getMaterialByName(matlname);
    if(!result){ 
      throw ProblemSetupException("Cannot find a material named:"+matlname, __FILE__, __LINE__);
    }
  }
  return result;
}

void SimulationState::clearStats()
{
  compilationTime = 0;
  regriddingTime = 0;
  regriddingCompilationTime = 0;
  regriddingCopyDataTime = 0;
  loadbalancerTime = 0;
  taskExecTime = 0;
  taskGlobalCommTime = 0;
  taskLocalCommTime = 0;
  taskWaitCommTime = 0;
  taskWaitThreadTime = 0;
  outputTime = 0;
}

void SimulationState::setDimensionality(bool x, bool y, bool z)
{
  d_numDims = 0;
  int currentDim = 0;
  bool args[3] = {x,y,z};

  for (int i = 0; i < 3; i++) {
    if (args[i]) {
      d_numDims++;
      d_activeDims[currentDim] = i;
      currentDim++;
    }
  }
}
