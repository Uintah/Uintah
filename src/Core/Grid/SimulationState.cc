/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <Core/Grid/Material.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <sci_defs/uintah_defs.h>

#ifndef NO_ARCHES
#include <CCA/Components/Arches/ArchesMaterial.h>
#endif

#ifndef NO_FVM
#include <CCA/Components/FVM/FVMMaterial.h>
#endif

#ifndef NO_ICE
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#endif

#ifndef NO_MPM
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#endif

#ifndef NO_WASATCH
#include <CCA/Components/Wasatch/WasatchMaterial.h>
#endif

using namespace Uintah;

int SimulationState::count = 0;  

SimulationState::SimulationState()
{
  if (count++ >= 1) {
    throw ProblemSetupException("Allocated multiple SimulationStates", __FILE__, __LINE__);
  }
}

//__________________________________
//
SimulationState::~SimulationState()
{
  clearMaterials();

  for (unsigned i = 0; i < old_matls.size(); i++){
    delete old_matls[i];
  }

  if(orig_all_matls && orig_all_matls->removeReference()){
    delete orig_all_matls;
  }
}

//__________________________________
//
void
SimulationState::registerMaterial( Material* matl )
{
  matl->registerParticleState(this);        
  matl->setDWIndex((int)matls.size());      

  matls.push_back(matl);                    

  if(matl->hasName()) {                    
    named_matls[matl->getName()] = matl;
  }
}

//__________________________________
//
void
SimulationState::registerMaterial( Material* matl,unsigned int index )
{
  matl->registerParticleState(this);
  matl->setDWIndex(index);

  if (matls.size() <= index) {
    matls.resize(index + 1);
  }
  matls[index] = matl;

  if (matl->hasName()) {
    named_matls[matl->getName()] = matl;
  }
}

//__________________________________
//
void
SimulationState::registerSimpleMaterial( SimpleMaterial* matl )
{
  simple_matls.push_back(matl);
  registerMaterial(matl);
}

//__________________________________
//
const MaterialSet*
SimulationState::allMaterials() const
{
  ASSERT(all_matls != nullptr);
  return all_matls;
}

//__________________________________
//
const MaterialSet*
SimulationState::originalAllMaterials() const
{
  ASSERT(orig_all_matls != nullptr);
  return orig_all_matls;
}

//__________________________________
//
void
SimulationState::setOriginalMatlsFromRestart( MaterialSet* matls )
{
  if (orig_all_matls && orig_all_matls->removeReference())
    delete orig_all_matls;
  orig_all_matls = matls;
}

//__________________________________
//
Material*
SimulationState::getMaterialByName( const std::string& name ) const
{
  std::map<std::string, Material*>::const_iterator iter = named_matls.find(name);
  if(iter == named_matls.end()){
    return nullptr;
  }
  return iter->second;
}

//__________________________________
//
Material*
SimulationState::parseAndLookupMaterial( ProblemSpecP& params, const std::string& name ) const
{
  // for single material problems return matl 0
  Material* result = getMaterial(0);

  if (getNumMatls() > 1) {
    std::string matlname;
    if (!params->get(name, matlname)) {
      throw ProblemSetupException("Cannot find material section", __FILE__, __LINE__);
    }

    result = getMaterialByName(matlname);
    if (!result) {
      throw ProblemSetupException("Cannot find a material named:" + matlname, __FILE__, __LINE__);
    }
  }
  return result;
}


//__________________________________
//
#ifndef NO_ARCHES
void
SimulationState::registerArchesMaterial( ArchesMaterial* matl )
{
   arches_matls.push_back(matl);
   registerMaterial(matl);
}
const MaterialSet* SimulationState::allArchesMaterials() const
{
  ASSERT(all_arches_matls != nullptr);
  return all_arches_matls;
}
#endif
//__________________________________
//
#ifndef NO_FVM
void
SimulationState::registerFVMMaterial( FVMMaterial* matl )
{
  fvm_matls.push_back(matl);
  registerMaterial(matl);
}

void
SimulationState::registerFVMMaterial( FVMMaterial* matl,unsigned int index )
{
  fvm_matls.push_back(matl);
  registerMaterial(matl,index);
}

const MaterialSet*
SimulationState::allFVMMaterials() const
{
  ASSERT(all_fvm_matls != nullptr);
  return all_fvm_matls;
}
#endif
//__________________________________
//
#ifndef NO_ICE
void
SimulationState::registerICEMaterial( ICEMaterial* matl )
{
   ice_matls.push_back(matl);
   registerMaterial(matl);
}

void
SimulationState::registerICEMaterial( ICEMaterial* matl,unsigned int index )
{
   ice_matls.push_back(matl);
   registerMaterial(matl,index);
}

const MaterialSet* SimulationState::allICEMaterials() const
{
  ASSERT(all_ice_matls != nullptr);
  return all_ice_matls;
}
#endif

//__________________________________
//
#ifndef NO_MPM
void
SimulationState::registerCZMaterial( CZMaterial* matl )
{
  cz_matls.push_back(matl);
  registerMaterial(matl);
}

//__________________________________
//
void
SimulationState::registerCZMaterial( CZMaterial* matl,unsigned int index )
{
  cz_matls.push_back(matl);
  registerMaterial(matl,index);
}

const MaterialSet*
SimulationState::allCZMaterials() const
{
  ASSERT(all_cz_matls != nullptr);
  return all_cz_matls;
}

//__________________________________
//
void
SimulationState::registerMPMMaterial( MPMMaterial* matl )
{
  mpm_matls.push_back(matl);
  registerMaterial(matl);
}

//__________________________________
//
void
SimulationState::registerMPMMaterial( MPMMaterial* matl,unsigned int index )
{
  mpm_matls.push_back(matl);
  registerMaterial(matl,index);
}

//__________________________________
//
const MaterialSet*
SimulationState::allMPMMaterials() const
{
  ASSERT(all_mpm_matls != nullptr);
  return all_mpm_matls;
}
#endif

//__________________________________
//
#ifndef NO_WASATCH
void
SimulationState::registerWasatchMaterial( WasatchMaterial* matl )
{
  wasatch_matls.push_back(matl);
  registerMaterial(matl);
}

//__________________________________
//
void
SimulationState::registerWasatchMaterial( WasatchMaterial* matl,unsigned int index )
{
  wasatch_matls.push_back(matl);
  registerMaterial(matl,index);
}

//__________________________________
//
const MaterialSet*
SimulationState::allWasatchMaterials() const
{
  ASSERT(all_wasatch_matls != nullptr);
  return all_wasatch_matls;
}
#endif

//__________________________________
//
void
SimulationState::finalizeMaterials()
{
  // All Matls
  if (all_matls && all_matls->removeReference()) {
    delete all_matls;
  }
  all_matls = scinew MaterialSet();
  all_matls->addReference();
  std::vector<int> tmp_matls(matls.size());

  for (size_t i = 0; i < matls.size(); i++) {
    tmp_matls[i] = matls[i]->getDWIndex();
  }
  all_matls->addAll(tmp_matls);

  // Original All Matls
  if (orig_all_matls == nullptr) {
    orig_all_matls = scinew MaterialSet();
    orig_all_matls->addReference();
    orig_all_matls->addAll(tmp_matls);
  }

  // All In One Matls
  if (allInOneMatl && allInOneMatl->removeReference()) {
    delete allInOneMatl;
  }
  allInOneMatl = scinew MaterialSubset();
  allInOneMatl->addReference();
  // a material that represents all materials
  // (i.e. summed over all materials -- the whole enchilada)
  allInOneMatl->add((int)matls.size());

  // Arches Matls
#ifndef NO_ARCHES
  if (all_arches_matls && all_arches_matls->removeReference()) {
    delete all_arches_matls;
  }
  all_arches_matls = scinew MaterialSet();
  all_arches_matls->addReference();
  std::vector<int> tmp_arches_matls(arches_matls.size());
  for (size_t i = 0; i < arches_matls.size(); i++) {
    tmp_arches_matls[i] = arches_matls[i]->getDWIndex();
  }
  all_arches_matls->addAll(tmp_arches_matls);
#endif

  // FVM Matls
#ifndef NO_FVM
  if (all_fvm_matls && all_fvm_matls->removeReference()){
    delete all_fvm_matls;
  }
  all_fvm_matls = scinew MaterialSet();
  all_fvm_matls->addReference();
  std::vector<int> tmp_fvm_matls(fvm_matls.size());
  for(int i=0;i<(int)fvm_matls.size();i++) {
    tmp_fvm_matls[i] = fvm_matls[i]->getDWIndex();
  }
  all_fvm_matls->addAll(tmp_fvm_matls);
#endif

  // ICE Matls
#ifndef NO_ICE
  if (all_ice_matls && all_ice_matls->removeReference()) {
    delete all_ice_matls;
  }
  all_ice_matls = scinew MaterialSet();
  all_ice_matls->addReference();
  std::vector<int> tmp_ice_matls(ice_matls.size());
  for (size_t i = 0; i < ice_matls.size(); i++) {
    tmp_ice_matls[i] = ice_matls[i]->getDWIndex();
  }
  all_ice_matls->addAll(tmp_ice_matls);
#endif

  // MPM Matls
#ifndef NO_MPM
  // Cohesive Zone
  if (all_cz_matls && all_cz_matls->removeReference()) {
    delete all_cz_matls;
  }
  all_cz_matls = scinew MaterialSet();
  all_cz_matls->addReference();
  std::vector<int> tmp_cz_matls(cz_matls.size());
  for (size_t i = 0; i < cz_matls.size(); i++) {
    tmp_cz_matls[i] = cz_matls[i]->getDWIndex();
  }
  all_cz_matls->addAll(tmp_cz_matls);

  // MPM
  if (all_mpm_matls && all_mpm_matls->removeReference()) {
    delete all_mpm_matls;
  }

  all_mpm_matls = scinew MaterialSet();
  all_mpm_matls->addReference();
  std::vector<int> tmp_mpm_matls(mpm_matls.size());
  for (size_t i = 0; i < mpm_matls.size(); i++) {
    tmp_mpm_matls[i] = mpm_matls[i]->getDWIndex();
  }
  all_mpm_matls->addAll(tmp_mpm_matls);
#endif

  // Wasatch Matls
#ifndef NO_WASATCH
  if (all_wasatch_matls && all_wasatch_matls->removeReference()) {
    delete all_wasatch_matls;
  }
  all_wasatch_matls = scinew MaterialSet();
  all_wasatch_matls->addReference();
  std::vector<int> tmp_wasatch_matls(wasatch_matls.size());

  for (size_t i = 0; i < wasatch_matls.size(); i++) {
    tmp_wasatch_matls[i] = wasatch_matls[i]->getDWIndex();
  }
  all_wasatch_matls->addAll(tmp_wasatch_matls);
#endif
}

//__________________________________
//
void
SimulationState::clearMaterials()
{
  for (size_t i = 0; i < matls.size(); i++){
    old_matls.push_back(matls[i]);
  }

  if(all_matls && all_matls->removeReference()){
    delete all_matls;
  }
  all_matls = nullptr;
  
  if (allInOneMatl && allInOneMatl->removeReference()) {
    delete allInOneMatl;
  }
  allInOneMatl = nullptr;

  matls.clear();
  named_matls.clear();
  simple_matls.clear();
  
#ifndef NO_ARCHES
  if (all_arches_matls && all_arches_matls->removeReference()){
    delete all_arches_matls;
  }
  arches_matls.clear();
  all_arches_matls = nullptr;
#endif

#ifndef NO_FVM
  if(all_fvm_matls && all_fvm_matls->removeReference()){
    delete all_fvm_matls;
  }
  fvm_matls.clear();
  all_fvm_matls = nullptr;
#endif

#ifndef NO_ICE
  if(all_ice_matls && all_ice_matls->removeReference()){
    delete all_ice_matls;
  }
  ice_matls.clear();
  all_ice_matls = nullptr;
#endif
  
#ifndef NO_MPM
  if(all_cz_matls && all_cz_matls->removeReference()){
    delete all_cz_matls;
  }
  cz_matls.clear();
  all_cz_matls = nullptr;

  if(all_mpm_matls && all_mpm_matls->removeReference()){
    delete all_mpm_matls;
  }
  mpm_matls.clear();
  all_mpm_matls = nullptr;
  
  d_cohesiveZoneState.clear();
  d_cohesiveZoneState_preReloc.clear();

  d_particleState.clear();
  d_particleState_preReloc.clear();
#endif
  
#ifndef NO_WASATCH
  if(all_wasatch_matls && all_wasatch_matls->removeReference()){
    delete all_wasatch_matls;
  }
  wasatch_matls.clear();
  all_wasatch_matls = nullptr;
#endif
}
