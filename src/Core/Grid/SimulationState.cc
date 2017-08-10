/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/SimulationState.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/StringUtil.h>

#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/Wasatch/WasatchMaterial.h>
#include <CCA/Components/FVM/FVMMaterial.h>

using namespace Uintah;
using namespace std;

SimulationState::SimulationState(ProblemSpecP &ps)
{
  VarLabel* nonconstDelt = 
    VarLabel::create("delT", delt_vartype::getTypeDescription() );

  nonconstDelt->allowMultipleComputes();
  delt_label = nonconstDelt;

  refineFlag_label      = VarLabel::create("refineFlag",     CCVariable<int>::getTypeDescription());
  oldRefineFlag_label   = VarLabel::create("oldRefineFlag",  CCVariable<int>::getTypeDescription());
  refinePatchFlag_label = VarLabel::create("refinePatchFlag",PerPatch<int>::getTypeDescription());
  switch_label          = VarLabel::create("switchFlag",     max_vartype::getTypeDescription());

  //__________________________________
  //  These variables can be modified by a component.
  VarLabel* nonconstOutputInv =             // output interval
    VarLabel::create("outputInterval",
		     min_vartype::getTypeDescription() );
  VarLabel* nonconstOutputTimestepInv =     // output timestep interval
    VarLabel::create("outputTimestepInterval",
		     min_vartype::getTypeDescription() );

  VarLabel* nonconstCheckpointInv =         // check point interval
    VarLabel::create("checkpointInterval",
		     min_vartype::getTypeDescription() );
  
  VarLabel* nonconstCheckpointTimestepInv = // check point timestep interval
    VarLabel::create("checkpointTimestepInterval",
		     min_vartype::getTypeDescription() );

  nonconstOutputInv->allowMultipleComputes();
  nonconstOutputTimestepInv->allowMultipleComputes();

  nonconstCheckpointInv->allowMultipleComputes();
  nonconstCheckpointTimestepInv->allowMultipleComputes();

  outputInterval_label             = nonconstOutputInv;
  outputTimestepInterval_label     = nonconstOutputTimestepInv;

  checkpointInterval_label         = nonconstCheckpointInv;
  checkpointTimestepInterval_label = nonconstCheckpointTimestepInv;

  //__________________________________
  max_matl_index    = 0;

  all_mpm_matls     = 0;
  all_cz_matls      = 0;
  all_ice_matls     = 0;
  all_wasatch_matls = 0;  
  all_arches_matls  = 0;
  all_fvm_matls     = 0;
  all_matls         = 0;
  
  orig_all_matls    = 0;
  refine_flag_matls = 0;
  allInOneMatl      = 0;

  d_topLevelTimeStep = 0;
  d_elapsed_sim_time  = 0.0;
  d_elapsed_wall_time = 0.0;

  d_numDims = 0;
  d_activeDims[0] = d_activeDims[1] = d_activeDims[2] = 0;

  d_isCopyDataTimestep        = false;

  d_isRegridTimestep          = false;
   
  d_timeRefinementRatio       = 0;
  
  d_lockstepAMR               = false;
  d_updateOutputInterval      = false;
  d_updateCheckpointInterval  = false;
  d_recompileTaskGraph        = false;
  d_overheadAvg               = 0; // Average time spent in overhead.
  d_usingLocalFileSystems     = false;

  d_switchState               = false;
  d_haveModifiedVars          = false;
  
  d_simulationTime            = 0;

  d_maybeLast                 = false;

  ProblemSpecP amr = ps->findBlock("AMR");
  
  if (amr){
    amr->get("useLockStep", d_lockstepAMR);
  }
    
  std::string timeStr("seconds");
  std::string bytesStr("MBytes");
    
  d_runTimeStats.insert( CompilationTime,           std::string("Compilation"),           timeStr, 0 );
  d_runTimeStats.insert( RegriddingTime,            std::string("Regridding"),            timeStr, 0 );
  d_runTimeStats.insert( RegriddingCompilationTime, std::string("RegriddingCompilation"), timeStr, 0 );
  d_runTimeStats.insert( RegriddingCopyDataTime,    std::string("RegriddingCopyData"),    timeStr, 0 );
  d_runTimeStats.insert( LoadBalancerTime,          std::string("LoadBalancer"),          timeStr, 0 );

  d_runTimeStats.insert( TaskExecTime,       std::string("TaskExec"),           timeStr, 0 );
  d_runTimeStats.insert( TaskLocalCommTime,  std::string("TaskLocalComm"),      timeStr, 0 );
  d_runTimeStats.insert( TaskWaitCommTime,   std::string("TaskWaitCommTime"),   timeStr, 0 );
  d_runTimeStats.insert( TaskReduceCommTime, std::string("TaskReduceCommTime"), timeStr, 0 );
  d_runTimeStats.insert( TaskWaitThreadTime, std::string("TaskWaitThread"),     timeStr, 0 );

  d_runTimeStats.insert( XMLIOTime,          std::string("XMLIO"),            timeStr, 0 );
  d_runTimeStats.insert( OutputIOTime,       std::string("OutputIO"),         timeStr, 0 );
  d_runTimeStats.insert( ReductionIOTime,    std::string("ReductionIO"),      timeStr, 0 );
  d_runTimeStats.insert( CheckpointIOTime,   std::string("CheckpointIO"),     timeStr, 0 );
  d_runTimeStats.insert( CheckpointReductionIOTime, std::string("CheckpointReductionIO"),     timeStr, 0 );
  d_runTimeStats.insert( TotalIOTime,        std::string("TotalIO"),          timeStr, 0 );

  d_runTimeStats.insert( OutputIORate,       std::string("OutputIORate"),     "MBytes/sec", 0 );
  d_runTimeStats.insert( ReductionIORate,    std::string("ReductionIORate"),  "MBytes/sec", 0 );
  d_runTimeStats.insert( CheckpointIORate,   std::string("CheckpointIORate"), "MBytes/sec", 0 );
  d_runTimeStats.insert( CheckpointReducIORate, std::string("CheckpointReducIORate"), "MBytes/sec", 0 );

  d_runTimeStats.insert( SCIMemoryUsed,      std::string("SCIMemoryUsed"),      bytesStr, 0 );
  d_runTimeStats.insert( SCIMemoryMaxUsed,   std::string("SCIMemoryMaxUsed"),   bytesStr, 0 );
  d_runTimeStats.insert( SCIMemoryHighwater, std::string("SCIMemoryHighwater"), bytesStr, 0 );
  d_runTimeStats.insert( MemoryUsed,         std::string("MemoryUsed"),         bytesStr, 0 );
  d_runTimeStats.insert( MemoryResident,     std::string("MemoryResident"),     bytesStr, 0 );

#ifdef USE_PAPI_COUNTERS
  d_runTimeStats.insert( TotalFlops,  std::string("TotalFlops") , "FLOPS" , 0 );
  d_runTimeStats.insert( TotalVFlops, std::string("TotalVFlops"), "FLOPS" , 0 );
  d_runTimeStats.insert( L2Misses,    std::string("L2Misses")   , "misses", 0 );
  d_runTimeStats.insert( L3Misses,    std::string("L3Misses")   , "misses", 0 );
  d_runTimeStats.insert( TLBMisses,   std::string("TLBMisses")  , "misses", 0 );
#endif

  d_runTimeStats.validate( MAX_TIMING_STATS );

  resetStats();

#ifdef HAVE_VISIT
  d_doVisIt = false;
#endif
}
//__________________________________
//
void SimulationState::registerMaterial(Material* matl)
{
  matl->registerParticleState(this);        
  matl->setDWIndex((int)matls.size());      

  matls.push_back(matl);                    
  if ((int)matls.size() > max_matl_index) { 
    max_matl_index = matls.size();          
  }                                         

  if(matl->hasName()) {                    
    named_matls[matl->getName()] = matl;
  }
}
//__________________________________
//
void SimulationState::registerMaterial(Material* matl,unsigned int index)
{
  matl->registerParticleState(this);        
  matl->setDWIndex(index);                  

  if (matls.size() <= index){                
    matls.resize(index+1);
  }                  
  matls[index]=matl;                        

  if ((int)matls.size() > max_matl_index) { 
    max_matl_index = matls.size();          
  }                                         

  if(matl->hasName()){                      
    named_matls[matl->getName()] = matl;    
  }
}

//__________________________________
//
void SimulationState::registerMPMMaterial(MPMMaterial* matl)
{
  mpm_matls.push_back(matl);
  registerMaterial(matl);
}
//__________________________________
//
void SimulationState::registerMPMMaterial(MPMMaterial* matl,unsigned int index)
{
  mpm_matls.push_back(matl);
  registerMaterial(matl,index);
}
//__________________________________
//
void SimulationState::registerCZMaterial(CZMaterial* matl)
{
  cz_matls.push_back(matl);
  registerMaterial(matl);
}
//__________________________________
//
void SimulationState::registerCZMaterial(CZMaterial* matl,unsigned int index)
{
  cz_matls.push_back(matl);
  registerMaterial(matl,index);
}
//__________________________________
//
void SimulationState::registerArchesMaterial(ArchesMaterial* matl)
{
   arches_matls.push_back(matl);
   registerMaterial(matl);
}
//__________________________________
//
void SimulationState::registerICEMaterial(ICEMaterial* matl)
{
   ice_matls.push_back(matl);
   registerMaterial(matl);
}
//__________________________________
//
void SimulationState::registerICEMaterial(ICEMaterial* matl,unsigned int index)
{
   ice_matls.push_back(matl);
   registerMaterial(matl,index);
}
//__________________________________
//
void SimulationState::registerWasatchMaterial(WasatchMaterial* matl)
{
  wasatch_matls.push_back(matl);
  registerMaterial(matl);
}
//__________________________________
//
void SimulationState::registerWasatchMaterial(WasatchMaterial* matl,unsigned int index)
{
  wasatch_matls.push_back(matl);
  registerMaterial(matl,index);
}
//__________________________________
//
void SimulationState::registerFVMMaterial(FVMMaterial* matl)
{
  fvm_matls.push_back(matl);
  registerMaterial(matl);
}
//__________________________________
//
void SimulationState::registerFVMMaterial(FVMMaterial* matl,unsigned int index)
{
  fvm_matls.push_back(matl);
  registerMaterial(matl,index);
}
//__________________________________
//
void SimulationState::registerSimpleMaterial(SimpleMaterial* matl)
{
  simple_matls.push_back(matl);
  registerMaterial(matl);
}
//__________________________________
//
void SimulationState::finalizeMaterials()
{
                                    // MPM
  if (all_mpm_matls && all_mpm_matls->removeReference()){
    delete all_mpm_matls;
  }
  all_mpm_matls = scinew MaterialSet();
  all_mpm_matls->addReference();
  vector<int> tmp_mpm_matls(mpm_matls.size());
  for( int i=0; i<(int)mpm_matls.size(); i++ ) {
    tmp_mpm_matls[i] = mpm_matls[i]->getDWIndex();
  }
  all_mpm_matls->addAll(tmp_mpm_matls);
  
                                    // Cohesive Zone
  if (all_cz_matls && all_cz_matls->removeReference()){
    delete all_cz_matls;
  }
  all_cz_matls = scinew MaterialSet();
  all_cz_matls->addReference();
  vector<int> tmp_cz_matls(cz_matls.size());
  for( int i=0; i<(int)cz_matls.size(); i++ ) {
    tmp_cz_matls[i] = cz_matls[i]->getDWIndex();
  }
  all_cz_matls->addAll(tmp_cz_matls);
  
                                    // Arches Matls
  if (all_arches_matls && all_arches_matls->removeReference()){
    delete all_arches_matls;
  }
  all_arches_matls = scinew MaterialSet();
  all_arches_matls->addReference();
  vector<int> tmp_arches_matls(arches_matls.size());
  for (int i = 0; i<(int)arches_matls.size();i++){
    tmp_arches_matls[i] = arches_matls[i]->getDWIndex();
  }
  all_arches_matls->addAll(tmp_arches_matls);

                                    // ICE Matls
  if (all_ice_matls && all_ice_matls->removeReference()){
    delete all_ice_matls;
  }
  all_ice_matls = scinew MaterialSet();
  all_ice_matls->addReference();
  vector<int> tmp_ice_matls(ice_matls.size());
  for(int i=0;i<(int)ice_matls.size();i++) {
    tmp_ice_matls[i] = ice_matls[i]->getDWIndex();
  }
  all_ice_matls->addAll(tmp_ice_matls);

  // FVM Matls
  if (all_fvm_matls && all_fvm_matls->removeReference()){
    delete all_fvm_matls;
  }
  all_fvm_matls = scinew MaterialSet();
  all_fvm_matls->addReference();
  vector<int> tmp_fvm_matls(fvm_matls.size());
  for(int i=0;i<(int)fvm_matls.size();i++) {
    tmp_fvm_matls[i] = fvm_matls[i]->getDWIndex();
  }
  all_fvm_matls->addAll(tmp_fvm_matls);

                                    // Wasatch Matls
  if (all_wasatch_matls && all_wasatch_matls->removeReference()){
    delete all_wasatch_matls;
  }
  all_wasatch_matls = scinew MaterialSet();
  all_wasatch_matls->addReference();
  vector<int> tmp_wasatch_matls(wasatch_matls.size());

  for(int i=0;i<(int)wasatch_matls.size();i++){
    tmp_wasatch_matls[i] = wasatch_matls[i]->getDWIndex();
  }
  all_wasatch_matls->addAll(tmp_wasatch_matls);
  
                                      // All Matls
  if (all_matls && all_matls->removeReference()){
    delete all_matls;
  }
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

  if (allInOneMatl && allInOneMatl->removeReference()){
    delete allInOneMatl;
  }
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
//__________________________________
//
void SimulationState::clearMaterials()
{
  for (int i = 0; i < (int)matls.size(); i++){
    old_matls.push_back(matls[i]);
  }

  if(all_matls && all_matls->removeReference()){
    delete all_matls;
  }
  
  if(all_mpm_matls && all_mpm_matls->removeReference()){
    delete all_mpm_matls;
  }

  if(all_cz_matls && all_cz_matls->removeReference()){
    delete all_cz_matls;
  }

  if (all_arches_matls && all_arches_matls->removeReference()){
    delete all_arches_matls;
  }

  if(all_ice_matls && all_ice_matls->removeReference()){
    delete all_ice_matls;
  }
  
  if(all_fvm_matls && all_fvm_matls->removeReference()){
    delete all_fvm_matls;
  }

  if(all_wasatch_matls && all_wasatch_matls->removeReference()){
    delete all_wasatch_matls;
  }

  if (allInOneMatl && allInOneMatl->removeReference()) {
    delete allInOneMatl;
  }

  matls.clear();
  mpm_matls.clear();
  cz_matls.clear();
  arches_matls.clear();
  ice_matls.clear();
  wasatch_matls.clear();
  simple_matls.clear();
  fvm_matls.clear();
  named_matls.clear();
  d_particleState.clear();
  d_particleState_preReloc.clear();
  d_cohesiveZoneState.clear();
  d_cohesiveZoneState_preReloc.clear();

  all_matls         = 0;
  all_mpm_matls     = 0;
  all_cz_matls      = 0;
  all_arches_matls  = 0;
  all_ice_matls     = 0;
  all_fvm_matls     = 0;
  all_wasatch_matls = 0;
  allInOneMatl      = 0;
}
//__________________________________
//
SimulationState::~SimulationState()
{
  VarLabel::destroy(delt_label);
  VarLabel::destroy(refineFlag_label);
  VarLabel::destroy(oldRefineFlag_label);
  VarLabel::destroy(refinePatchFlag_label);
  VarLabel::destroy(switch_label);
  VarLabel::destroy(outputInterval_label);
  VarLabel::destroy(outputTimestepInterval_label);
  VarLabel::destroy(checkpointInterval_label);
  VarLabel::destroy(checkpointTimestepInterval_label);
  clearMaterials();

  for (unsigned i = 0; i < old_matls.size(); i++){
    delete old_matls[i];
  }

  if(refine_flag_matls && refine_flag_matls->removeReference()){
    delete refine_flag_matls;
  }

  if(orig_all_matls && orig_all_matls->removeReference()){
    delete orig_all_matls;
  }
}
//__________________________________
//
const MaterialSet* SimulationState::allMPMMaterials() const
{
  ASSERT(all_mpm_matls != 0);
  return all_mpm_matls;
}
//__________________________________
//
const MaterialSet* SimulationState::allCZMaterials() const
{
  ASSERT(all_cz_matls != 0);
  return all_cz_matls;
}
//__________________________________
//
const MaterialSet* SimulationState::allArchesMaterials() const
{
  ASSERT(all_arches_matls != 0);
  return all_arches_matls;
}
//__________________________________
//
const MaterialSet* SimulationState::allICEMaterials() const
{
  ASSERT(all_ice_matls != 0);
  return all_ice_matls;
}
//__________________________________
//
const MaterialSet* SimulationState::allFVMMaterials() const
{
  ASSERT(all_fvm_matls != 0);
  return all_fvm_matls;
}
//__________________________________
//
const MaterialSet* SimulationState::allWasatchMaterials() const
{
  ASSERT(all_wasatch_matls != 0);
  return all_wasatch_matls;
}
//__________________________________
//
const MaterialSet* SimulationState::allMaterials() const
{
  ASSERT(all_matls != 0);
  return all_matls;
}
//__________________________________
//
const MaterialSet* SimulationState::originalAllMaterials() const
{
  ASSERT(orig_all_matls != 0);
  return orig_all_matls;
}
//__________________________________
//
void SimulationState::setOriginalMatlsFromRestart(MaterialSet* matls)
{
  if (orig_all_matls && orig_all_matls->removeReference())
    delete orig_all_matls;
  orig_all_matls = matls;
}
  
//__________________________________
//
const MaterialSubset* SimulationState::refineFlagMaterials() const
{
  ASSERT(refine_flag_matls != 0);
  return refine_flag_matls;
}
//__________________________________
//
Material* SimulationState::getMaterialByName(const std::string& name) const
{
  map<string, Material*>::const_iterator iter = named_matls.find(name);
  if(iter == named_matls.end()){
    return 0;
  }
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
//__________________________________
//
void SimulationState::resetStats()
{
  d_runTimeStats.reset( 0 );  
  d_otherStats.reset( 0 );  
}
//__________________________________
//
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

