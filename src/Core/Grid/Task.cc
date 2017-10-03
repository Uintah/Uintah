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

#include <Core/Grid/Task.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/StringUtil.h>


#include <set>

using namespace Uintah;

MaterialSubset* Task::globalMatlSubset = nullptr;


//______________________________________________________________________
//
Task::~Task()
{
  if (m_action) {
    delete m_action;
  }

  Dependency* dep = m_req_head;
  while (dep) {
    Dependency* next = dep->m_next;
    delete dep;
    dep = next;
  }

  dep = m_comp_head;
  while (dep) {
    Dependency* next = dep->m_next;
    delete dep;
    dep = next;
  }

  dep = m_mod_head;
  while (dep) {
    Dependency* next = dep->m_next;
    delete dep;
    dep = next;
  }

  if (m_matl_set && m_matl_set->removeReference()) {
    delete m_matl_set;
  }

  if (m_patch_set && m_patch_set->removeReference()) {
    delete m_patch_set;
  }

  // easier to periodically delete this than to force a call to a cleanup
  // function, and probably not very expensive.
  if (globalMatlSubset && globalMatlSubset->removeReference()) {
    delete globalMatlSubset;
  }

  globalMatlSubset = nullptr;
}

//______________________________________________________________________

//
void
Task::initialize()
{
  m_comp_head = nullptr;
  m_comp_tail = nullptr;
  m_req_head  = nullptr;
  m_req_tail  = nullptr;
  m_mod_head  = nullptr;
  m_mod_tail  = nullptr;
  m_patch_set = nullptr;
  m_matl_set  = nullptr;

  m_uses_mpi         = false;
  m_uses_threads     = false;
  m_uses_device      = false;
  m_subpatch_capable = false;
  m_has_subscheduler = false;

  for (int i = 0; i < TotalDWs; i++) {
    m_dwmap[i] = Task::InvalidDW;
  }

  m_sorted_order = -1;
  m_phase        = -1;
  m_comm         = -1;

  //The 0th level has a max ghost cell of zero.  Other levels are left uninitialized.
  m_max_ghost_cells[0] = 0;
  m_max_level_offset   = 0;
}

//______________________________________________________________________

//
void
Task::setSets( const PatchSet* ps, const MaterialSet* ms )
{
  // NOTE: the outer [patch/matl]Set checks are related to temporal scheduling, e.g. more then 1 regular task graph
  //
  // This is called from TaskGraph::addTask() in which a single task may be added to >1 Normal
  // task graph. In this case, first time here, m_path/matl_set will be nullptr and subsequent visits
  // will be the same pointer as ps and ms respectively. Without these checks, the refCount gets
  // artificially inflated and ComputeSubsets (Patch/Matl)) are not deleted - mem leak. APH, 06/08/17
  if (m_patch_set == nullptr) {
    m_patch_set = ps;
    if (m_patch_set) {
      m_patch_set->addReference();
    }
  }

  if (m_matl_set == nullptr) {
    m_matl_set = ms;
    if (m_matl_set) {
      m_matl_set->addReference();
    }
  }
}

//______________________________________________________________________

//
const MaterialSubset*
Task::getGlobalMatlSubset()
{
  if (globalMatlSubset == nullptr) {
    globalMatlSubset = scinew MaterialSubset();
    globalMatlSubset->add(-1);
    globalMatlSubset->addReference();
  }
  return globalMatlSubset;
}

//______________________________________________________________________

//
void
Task::usesMPI(bool state)
{
  m_uses_mpi = state;
}

//______________________________________________________________________

//
void
Task::hasSubScheduler(bool state)
{
  m_has_subscheduler = state;
}

//______________________________________________________________________

//
void
Task::usesThreads(bool state)
{
  m_uses_threads = state;
}

//______________________________________________________________________
//
void
Task::usesDevice(bool state, int maxStreamsPerTask /* = 1 */ )
{
  m_uses_device = state;
  m_max_streams_per_task = maxStreamsPerTask;
}

//______________________________________________________________________
//
void Task::requires(       WhichDW             dw
                   , const VarLabel          * var
                   , const PatchSubset       * patches
                   ,       PatchDomainSpec     patches_dom
                   ,       int                 level_offset
                   , const MaterialSubset    * matls
                   ,      MaterialDomainSpec   matls_dom
                   ,      Ghost::GhostType     gtype
                   ,      int                  numGhostCells
                   ,      bool                 oldTG
                   )
{
  if (matls == nullptr && var->typeDescription()->isReductionVariable()) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  }
  else if (matls != nullptr && matls->size() == 0) {
    return;  // no materials, no dependency
  }

  Dependency* dep = scinew Dependency(Requires, this, dw, var, oldTG, patches, matls, patches_dom,
                                      matls_dom, gtype, numGhostCells, level_offset);

  if (level_offset > m_max_level_offset) {
    m_max_level_offset = level_offset;
  }

  dep->m_next = nullptr;
  if (m_req_tail) {
    m_req_tail->m_next = dep;
  }
  else {
    m_req_head = dep;
  }
  m_req_tail = dep;

  if (dw == OldDW) {
    m_requires_old_dw.insert(std::make_pair(var, dep));
  }
  else {
    m_requires.insert(std::make_pair(var, dep));
  }
}

//______________________________________________________________________
//
void Task::requires(       WhichDW              dw
                   , const VarLabel           * var
                   , const PatchSubset        * patches
                   ,       PatchDomainSpec      patches_dom
                   , const MaterialSubset     * matls
                   ,       MaterialDomainSpec   matls_dom
                   ,       Ghost::GhostType     gtype
                   ,       int                  numGhostCells
                   ,       bool                 oldTG
                   )
{
  int offset = 0;
  if (patches_dom == CoarseLevel || patches_dom == FineLevel) {
    offset = 1;
  }
  requires(dw, var, patches, patches_dom, offset, matls, matls_dom, gtype, numGhostCells, oldTG);
}

//______________________________________________________________________
//
void Task::requires(WhichDW dw,
                    const VarLabel* var,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    Ghost::GhostType gtype,
                    int numGhostCells,
                    bool oldTG)
{
  requires(dw, var, patches, ThisLevel, matls, NormalDomain, gtype, numGhostCells, oldTG);
}

//______________________________________________________________________
//
void
Task::requires(       WhichDW            dw
              , const VarLabel         * var
              ,       Ghost::GhostType   gtype
              ,       int                numGhostCells
              ,       bool               oldTG
              )
{
  requires(dw, var, nullptr, ThisLevel, nullptr, NormalDomain, gtype, numGhostCells, oldTG);
}

//______________________________________________________________________
//
void
Task::requires(       WhichDW            dw
              , const VarLabel         * var
              , const MaterialSubset   * matls
              ,       Ghost::GhostType   gtype
              ,       int                numGhostCells
              ,       bool               oldTG
              )
{
  requires(dw, var, nullptr, ThisLevel, matls, NormalDomain, gtype, numGhostCells, oldTG);
}

//______________________________________________________________________
//
void
Task::requires(       WhichDW              dw
              , const VarLabel           * var
              , const MaterialSubset     * matls
              ,       MaterialDomainSpec   matls_dom
              ,       Ghost::GhostType     gtype
              ,       int                  numGhostCells
              ,       bool                 oldTG
              )
{
  requires(dw, var, nullptr, ThisLevel, matls, matls_dom, gtype, numGhostCells, oldTG);
}

//______________________________________________________________________
//
void
Task::requires( WhichDW                  dw
              , const VarLabel         * var
              , const PatchSubset      * patches
              ,       Ghost::GhostType   gtype
              ,       int                numGhostCells
              ,       bool               oldTG
              )
{
  requires(dw, var, patches, ThisLevel, nullptr, NormalDomain, gtype, numGhostCells, oldTG);
}

//______________________________________________________________________
//
void
Task::requires( WhichDW                dw
              , const VarLabel       * var
              , const PatchSubset    * patches
              , const MaterialSubset * matls
              )
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if (vartype == TypeDescription::SoleVariable) {
    requires(dw, var, (const Level*)0, matls);
  }
  else if (vartype == TypeDescription::PerPatch) {
    requires(dw, var, patches, ThisLevel, matls, NormalDomain, Ghost::None, 0);
  }
  else {
    SCI_THROW(InternalError("Requires should specify ghost type or level for this variable", __FILE__, __LINE__));
  }
}

//______________________________________________________________________
//
void
Task::requires(       WhichDW          dw
              , const VarLabel       * var
              , const MaterialSubset * matls
              ,       bool             oldTG
              )
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if(!(vartype == TypeDescription::PerPatch
       || vartype == TypeDescription::ReductionVariable
       || vartype == TypeDescription::SoleVariable)) {
    SCI_THROW(InternalError("Requires should specify ghost type for this variable", __FILE__, __LINE__));
  }
  
  if(vartype == TypeDescription::ReductionVariable) {
    requires(dw, var, (const Level*)0, matls, NormalDomain, oldTG);
  } else if(vartype == TypeDescription::SoleVariable) {
    requires(dw, var, (const Level*)0, matls);
  } else {
    requires(dw, var, nullptr, ThisLevel, matls, NormalDomain, Ghost::None, 0, oldTG);
  }
}

//______________________________________________________________________
//
void Task::requires(       WhichDW              dw
                   , const VarLabel           * var
                   , const Level              * level
                   , const MaterialSubset     * matls
                   ,       MaterialDomainSpec   matls_dom
                   ,       bool                 oldTG
                   )
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if (!(vartype == TypeDescription::ReductionVariable || vartype == TypeDescription::SoleVariable)) {
    SCI_THROW(InternalError("Requires should specify ghost type for this variable", __FILE__, __LINE__));
  }

  if (matls == nullptr) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  }
  else if (matls->size() == 0) {
    return;  // no materials, no dependency
  }

  Dependency* dep = scinew Dependency(Requires, this, dw, var, oldTG, level, matls, matls_dom);
  dep->m_next = nullptr;

  if (m_req_tail) {
    m_req_tail->m_next = dep;
  }
  else {
    m_req_head = dep;
  }
  m_req_tail = dep;

  if (dw == OldDW) {
    m_requires_old_dw.insert(std::make_pair(var, dep));
  }
  else {
    m_requires.insert(std::make_pair(var, dep));
  }
}

//______________________________________________________________________
//
void Task::computes( const VarLabel           * var
                   , const PatchSubset        * patches
                   ,       PatchDomainSpec      patches_dom
                   , const MaterialSubset     * matls
                   ,       MaterialDomainSpec   matls_dom
                   )
{
  if (var->typeDescription()->isReductionVariable()) {
    if (matls == nullptr) {
      // default material for a reduction variable is the global material (-1)
      matls = getGlobalMatlSubset();
      matls_dom = OutOfDomain;
    }
    ASSERT(patches == nullptr);
  }

  Dependency* dep = scinew Dependency(Computes, this, NewDW, var, false, patches, matls, patches_dom, matls_dom);
  dep->m_next = nullptr;

  if (m_comp_tail) {
    m_comp_tail->m_next = dep;
  }
  else {
    m_comp_head = dep;
  }
  m_comp_tail = dep;

  m_computes.insert(std::make_pair(var, dep));
}

//______________________________________________________________________
//
void Task::computes( const VarLabel       * var
                   , const PatchSubset    * patches
                   , const MaterialSubset * matls
                   )
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if (vartype == TypeDescription::ReductionVariable || vartype == TypeDescription::SoleVariable) {
    computes(var, (const Level*)nullptr, matls);
  }
  else {
    computes(var, patches, ThisLevel, matls, NormalDomain);
  }
}

//______________________________________________________________________
//
void
Task::computes( const VarLabel * var, const MaterialSubset * matls )
{
  computes(var, nullptr, ThisLevel, matls, NormalDomain);
}

//______________________________________________________________________
//
void Task::computes( const VarLabel            * var
                   , const MaterialSubset      * matls
                   ,       MaterialDomainSpec    matls_dom
                   )
{
  computes(var, nullptr, ThisLevel, matls, matls_dom);
}

//______________________________________________________________________
//
void Task::computes( const VarLabel        * var
                   , const PatchSubset     * patches
                   ,       PatchDomainSpec   patches_dom
                   )
{
  computes(var, patches, patches_dom, nullptr, NormalDomain);
}

//______________________________________________________________________
//
void Task::computes( const VarLabel           * var
                   , const Level              * level
                   , const MaterialSubset     * matls
                   ,       MaterialDomainSpec   matls_dom
                   )
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if (!(vartype == TypeDescription::ReductionVariable || vartype == TypeDescription::SoleVariable)) {
    SCI_THROW(InternalError("Computes should only be used for reduction variable", __FILE__, __LINE__));
  }

  if (matls == nullptr) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  }
  else if (matls->size() == 0) {
    throw InternalError("Computes of an empty material set!", __FILE__, __LINE__);
  }

  Dependency* dep = scinew Dependency(Computes, this, NewDW, var, false, level, matls, matls_dom);
  dep->m_next = nullptr;

  if (m_comp_tail) {
    m_comp_tail->m_next = dep;
  }
  else {
    m_comp_head = dep;
  }
  m_comp_tail = dep;

  m_computes.insert(std::make_pair(var, dep));
}

//______________________________________________________________________
//
void Task::computesWithScratchGhost( const VarLabel           * var
                                   , const MaterialSubset     * matls
                                   ,       MaterialDomainSpec   matls_dom
                                   ,       Ghost::GhostType     gtype
                                   ,       int                  numGhostCells
                                   ,       bool                 oldTG
                                   )
{
  if (var->typeDescription()->isReductionVariable()) {
    SCI_THROW(InternalError("ComputeswithScratchGhost should not be used for reduction variable", __FILE__, __LINE__));
  }

  Dependency* dep = scinew Dependency(Computes, this, NewDW, var, oldTG, nullptr, matls,
                                      ThisLevel, matls_dom, gtype, numGhostCells);
  dep->m_next = nullptr;

  if (m_comp_tail) {
    m_comp_tail->m_next = dep;
  }
  else {
    m_comp_head = dep;
  }

  m_comp_tail = dep;

  m_computes.insert(std::make_pair(var, dep));
}


//______________________________________________________________________
//
void Task::modifiesWithScratchGhost( const VarLabel           * var
                                   , const PatchSubset        * patches
                                   ,       PatchDomainSpec      patches_dom
                                   , const MaterialSubset     * matls
                                   ,       MaterialDomainSpec   matls_dom
                                   ,       Ghost::GhostType     gtype
                                   ,       int                  numGhostCells
                                   ,       bool                 oldTG
                                   )
{
  this->requires(NewDW, var, patches, patches_dom, matls, matls_dom, gtype, numGhostCells);
  this->modifies(var, patches, patches_dom, matls, matls_dom);
}

//______________________________________________________________________
//
void Task::modifies( const VarLabel           * var
                   , const PatchSubset        * patches
                   ,       PatchDomainSpec      patches_dom
                   , const MaterialSubset     * matls
                   ,       MaterialDomainSpec   matls_dom
                   ,       bool                 oldTG
                   )
{
  if (matls == nullptr && var->typeDescription()->isReductionVariable()) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
    ASSERT(patches == nullptr);
  }

  Dependency* dep = scinew Dependency(Modifies, this, NewDW, var, oldTG, patches, matls, patches_dom, matls_dom);
  dep->m_next = nullptr;
  if (m_mod_tail) {
    m_mod_tail->m_next = dep;
  }
  else {
    m_mod_head = dep;
  }
  m_mod_tail = dep;

  m_requires.insert(std::make_pair(var, dep));
  m_computes.insert(std::make_pair(var, dep));
  m_modifies.insert(std::make_pair(var, dep));
}

//______________________________________________________________________
//
void Task::modifies( const VarLabel       * var
                   , const Level          * level
                   , const MaterialSubset * matls
                   , MaterialDomainSpec     matls_domain
                   , bool                   oldTG
                   )
{
  const TypeDescription* vartype = var->typeDescription();

  if (matls == nullptr && vartype->isReductionVariable()) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_domain = OutOfDomain;
  }

  if (!vartype->isReductionVariable()) {
    SCI_THROW(InternalError("modifies with level should only be used for reduction variable", __FILE__, __LINE__));
  }

  Dependency* dep = scinew Dependency(Modifies, this, NewDW, var, oldTG, level, matls, matls_domain);
  dep->m_next = nullptr;
  if (m_mod_tail) {
    m_mod_tail->m_next = dep;
  }
  else {
    m_mod_head = dep;
  }
  m_mod_tail = dep;

  m_requires.insert(std::make_pair(var, dep));
  m_computes.insert(std::make_pair(var, dep));
  m_modifies.insert(std::make_pair(var, dep));
}

//______________________________________________________________________
//
void Task::modifies( const VarLabel       * var
                   , const PatchSubset    * patches
                   , const MaterialSubset * matls
                   ,       bool             oldTG
                   )
{
  modifies(var, patches, ThisLevel, matls, NormalDomain, oldTG);
}

//______________________________________________________________________
//
void
Task::modifies( const VarLabel * var, bool oldTG )
{
  modifies(var, nullptr, ThisLevel, nullptr, NormalDomain, oldTG);
}

//______________________________________________________________________
//
void
Task::modifies( const VarLabel       * var
              , const MaterialSubset * matls
              , bool                   oldTG
              )
{
  modifies(var, nullptr, ThisLevel, matls, NormalDomain, oldTG);
}

//______________________________________________________________________
//
void Task::modifies( const VarLabel           * var
                   , const MaterialSubset     * matls
                   ,       MaterialDomainSpec   matls_dom
                   ,       bool                 oldTG
                   )
{
  modifies(var, nullptr, ThisLevel, matls, matls_dom, oldTG);
}

//______________________________________________________________________
//
bool Task::hasComputes( const VarLabel * var
                      ,       int        matlIndex
                      , const Patch    * patch
                      ) const
{
  return isInDepMap(m_computes, var, matlIndex, patch);
}

//______________________________________________________________________
//
bool Task::hasRequires( const VarLabel  * var
                      ,       int         matlIndex
                      , const Patch     * patch
                      ,       IntVector   lowOffset
                      ,       IntVector   highOffset
                      ,       WhichDW     dw
                      ) const
{
  DepMap depMap = m_requires;
  
  if(dw == OldDW){
    depMap = m_requires_old_dw;
  }
  
  Dependency* dep = isInDepMap(depMap, var, matlIndex, patch);  
  
    
  if (dep) {
    // make sure we are within the allowed ghost cell limit
    IntVector allowableLowOffset, allowableHighOffset;
      
    Patch::getGhostOffsets(var->typeDescription()->getType(), dep->m_gtype, dep->m_num_ghost_cells,
                           allowableLowOffset, allowableHighOffset);
                           
    return ((Max(allowableLowOffset, lowOffset) == allowableLowOffset) &&
            (Max(allowableHighOffset, highOffset) == allowableHighOffset));
  }
  return false;
}

//______________________________________________________________________
//
bool Task::hasModifies( const VarLabel * var
                      , int              matlIndex
                      , const Patch    * patch
                      ) const
{
  return isInDepMap(m_modifies, var, matlIndex, patch);
}

//______________________________________________________________________
//
Task::Dependency*
Task::isInDepMap( const DepMap   & depMap
                , const VarLabel * var
                ,       int        matlIndex
                , const Patch    * patch
                ) const
{
  DepMap::const_iterator found_iter = depMap.find(var);

  // loop over dependency map and search for the right dependency
  while (found_iter != depMap.end() && (*found_iter).first->equals(var)) {

    Dependency* dep = (*found_iter).second;
    const PatchSubset* patches = dep->m_patches;
    const MaterialSubset* matls = dep->m_matls;

    bool hasPatches = false, hasMatls = false;

    if (patches == nullptr) {  // if patches == nullptr then the requirement for patches is satisfied
      hasPatches = true;
    }
    else {
      if (dep->m_patches_dom == Task::CoarseLevel) {  // check that the level of the patches matches the coarse level
        hasPatches = getLevel(getPatchSet())->getRelativeLevel(-dep->m_level_offset) == getLevel(patches);
      }
      else if (dep->m_patches_dom == Task::FineLevel) {  // check that the level of the patches matches the fine level
        hasPatches = getLevel(getPatchSet())->getRelativeLevel(dep->m_level_offset) == getLevel(patches);
      }
      else { // check that the patches subset contain the requested patch
        hasPatches = patches->contains(patch);
      }
    }
    if (matls == nullptr) { // if matls == nullptr then the requirement for matls is satisfied
      hasMatls = true;
    }
    else { // check that the malts subset contains the matlIndex
      hasMatls = matls->contains(matlIndex);
    }

    if (hasMatls && hasPatches) {  // if this dependency contains both the matls and patches return the dependency
      return dep;
    }
    found_iter++;
  }
  return nullptr;
}

//______________________________________________________________________
//
Task::Dependency::Dependency(       DepType              deptype
                            ,       Task               * task
                            ,       WhichDW              whichdw
                            , const VarLabel           * var
                            ,       bool                 oldTG
                            , const PatchSubset        * patches
                            , const MaterialSubset     * matls
                            ,       PatchDomainSpec      patches_dom
                            ,       MaterialDomainSpec   matls_dom
                            ,       Ghost::GhostType     gtype
                            ,       int                  numGhostCells
                            ,       int                  level_offset
                            )

    : m_dep_type(deptype)
    , m_task(task)
    , m_var(var)
    , m_look_in_old_tg(oldTG)
    , m_patches(patches)
    , m_matls(matls)
    , m_patches_dom(patches_dom)
    , m_matls_dom(matls_dom)
    , m_gtype(gtype)
    , m_whichdw(whichdw)
    , m_num_ghost_cells(numGhostCells)
    , m_level_offset(level_offset)
{
  if (var) {
    var->addReference();
  }

  if (patches) {
    patches->addReference();
  }

  if (matls) {
    matls->addReference();
  }
}

//______________________________________________________________________
//
Task::Dependency::Dependency(       DepType              deptype
                            ,       Task               * task
                            ,       WhichDW              whichdw
                            , const VarLabel           * var
                            ,       bool                 oldTG
                            , const Level              * reductionLevel
                            , const MaterialSubset     * matls
                            ,       MaterialDomainSpec   matls_dom
                            )

    : m_dep_type(deptype)
    , m_task(task)
    , m_var(var)
    , m_look_in_old_tg(oldTG)
    , m_matls(matls)
    , m_reduction_level(reductionLevel)
    , m_matls_dom(matls_dom)
    , m_gtype(Ghost::None)
    , m_whichdw(whichdw)
{
  if (var) {
    var->addReference();
  }

  if (matls) {
    matls->addReference();
  }
}

//______________________________________________________________________
//
Task::Dependency::~Dependency()
{
  VarLabel::destroy(m_var); // just remove the ref

  if(m_patches != nullptr && m_patches->removeReference()){
    delete m_patches;
  }
    
  if(m_matls != nullptr && m_matls->removeReference()){
    delete m_matls;
  }
}

// for xlC:
namespace Uintah {

//______________________________________________________________________
//
constHandle<PatchSubset>
Task::Dependency::getPatchesUnderDomain(const PatchSubset * domainPatches) const
{
  switch(m_patches_dom){
  case Task::ThisLevel:
  case Task::OtherGridDomain: // use the same patches, we'll figure out where it corresponds on the other grid
    return PatchSubset::intersection(m_patches, domainPatches);
  case Task::CoarseLevel:
  case Task::FineLevel:      
    return getOtherLevelPatchSubset(m_patches_dom, m_level_offset, m_patches, domainPatches, m_num_ghost_cells);
  default:
    SCI_THROW(InternalError(std::string("Unknown patch domain ") + " type " +
                            Uintah::to_string(static_cast<int>(m_patches_dom)), __FILE__, __LINE__));
  }
}

//______________________________________________________________________
//
constHandle<MaterialSubset>
Task::Dependency::getMaterialsUnderDomain(const MaterialSubset * domainMaterials) const
{
  switch(m_matls_dom){
  case Task::NormalDomain:
    return MaterialSubset::intersection(m_matls, domainMaterials);
  case Task::OutOfDomain:
    return m_matls;
  default:
    SCI_THROW(InternalError(std::string("Unknown matl domain ") + " type " +
                            Uintah::to_string(static_cast<int>(m_matls_dom)), __FILE__, __LINE__));
  }
}

//______________________________________________________________________
//
constHandle<PatchSubset> Task::Dependency::getOtherLevelPatchSubset(       Task::PatchDomainSpec   dom
                                                                   ,       int                     level_offset
                                                                   , const PatchSubset           * subset
                                                                   , const PatchSubset           * domainSubset
                                                                   ,       int                     ngc
                                                                   )
{
  constHandle<PatchSubset> myLevelSubset = PatchSubset::intersection(subset, domainSubset);

  int levelOffset = 0;
  switch (dom) {
    case Task::CoarseLevel :
      levelOffset = -level_offset;
      break;
    case Task::FineLevel :
      levelOffset = level_offset;
      break;
    default :
      SCI_THROW(InternalError("Unhandled DomainSpec in Task::Dependency::getOtherLevelComputeSubset", __FILE__, __LINE__));
  }

  std::set<const Patch*, Patch::Compare> patches;
  for (int p = 0; p < myLevelSubset->size(); p++) {
    const Patch* patch = myLevelSubset->get(p);
    Patch::selectType somePatches;
    patch->getOtherLevelPatches(levelOffset, somePatches, ngc);
    patches.insert(somePatches.begin(), somePatches.end());
  }

  return constHandle<PatchSubset>(scinew PatchSubset(patches.begin(), patches.end()));
}

} // end namespace Uintah

//______________________________________________________________________
//
void
Task::doit( DetailedTask                * task
          , CallBackEvent                 event
          , const ProcessorGroup        * pg
          , const PatchSubset           * patches
          , const MaterialSubset        * matls
          , std::vector<DataWarehouseP> & dws
          , void                        * oldTaskGpuDW
          , void                        * newTaskGpuDW
          , void                        * stream
          , int                           deviceID
          )
{
  DataWarehouse* fromDW = mapDataWarehouse(Task::OldDW, dws);
  DataWarehouse* toDW   = mapDataWarehouse(Task::NewDW, dws);

  if (m_action) {
    m_action->doit(task, event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID);
  }
}

//______________________________________________________________________
//
void
Task::display( std::ostream & out ) const
{
  out << Parallel::getMPIRank() << " " << getName();
  if (m_uses_device) {
    out << ": GPU task,";
  }

  out << " (" << d_tasktype << ")";

  if ( (d_tasktype == Task::Normal || d_tasktype == Task::Output ) && m_patch_set != nullptr) {
    out << ", Level " << getLevel(m_patch_set)->getIndex();
  }

  if (m_matl_set == nullptr) {
    out << ", No-Matl-Set";
  }
  else {
    out << ", " << *m_matl_set;
  }
  out << ", DWs: ";
  for (int i = 0; i < TotalDWs; i++) {
    if (i != 0) {
      out << ", ";
    }
    out << m_dwmap[i];
  }
  if (m_patch_set == nullptr) {
    out << ", No-Patch-Set";
  }
  else {
    out << ", " << *m_patch_set;
  }
}

//______________________________________________________________________
//
namespace Uintah {

std::ostream &
operator <<( std::ostream & out, const Uintah::Task::Dependency & dep )
{
  out << "[";
  out << std::left;
  out.width(20);
  out << *(dep.m_var) << ", ";

  // reduction variable
  if (dep.m_var->typeDescription()->isReductionVariable()) {
    if (dep.m_reduction_level) {
      out << " reduction Level: " << dep.m_reduction_level->getIndex();
    }
    else {
      out << " Global level";
    }
  }
  else {
    // all other variables:
    if (dep.m_patches) {
      out << " Level: " << getLevel(dep.m_patches)->getIndex();
      out << " Patches: ";
      for (int i = 0; i < dep.m_patches->size(); i++) {
        if (i > 0) {
          out << ",";
        }
        out << dep.m_patches->get(i)->getID();
      }
    }
    else if (dep.m_reduction_level) {
      out << " reduction Level: " << dep.m_reduction_level->getIndex();
    }
    else if (dep.m_patches_dom) {
      switch (dep.m_patches_dom) {
        case Task::CoarseLevel :
          out << "coarseLevel";
          break;
        case Task::FineLevel :
          out << "fineLevel";
          break;
        case Task::OtherGridDomain :
          out << "OtherGridDomain";
          break;
        case Task::ThisLevel :
          out << "ThisLevel";
          break;
        default :
          break;
      }
    }
    else {
      out << "all Patches";
    }
  }

  out << ", MI: ";
  if (dep.m_matls) {
    for (int i = 0; i < dep.m_matls->size(); i++) {
      if (i > 0) {
        out << ",";
      }
      out << dep.m_matls->get(i);
    }
  }
  else {
    out << "none";
  }
  out << ", ";
  switch (dep.m_whichdw) {
    case Task::OldDW :
      out << "OldDW";
      break;
    case Task::NewDW :
      out << "NewDW";
      break;
    case Task::CoarseOldDW :
      out << "CoarseOldDW";
      break;
    case Task::CoarseNewDW :
      out << "CoarseNewDW";
      break;
    case Task::ParentOldDW :
      out << "ParentOldDW";
      break;
    case Task::ParentNewDW :
      out << "ParentNewDW";
      break;
    default :
      out << "Unknown DW!";
      break;
  }
  out << " (mapped to dw index " << dep.m_task->mapDataWarehouse(dep.m_whichdw) << ")";
  out << ", ";
  switch (dep.m_gtype) {
    case Ghost::None :
      out << "Ghost::None";
      break;
    case Ghost::AroundNodes :
      out << "Ghost::AroundNodes";
      break;
    case Ghost::AroundCells :
      out << "Ghost::AroundCells";
      break;
    case Ghost::AroundFacesX :
      out << "Ghost::AroundFacesX";
      break;
    case Ghost::AroundFacesY :
      out << "Ghost::AroundFacesY";
      break;
    case Ghost::AroundFacesZ :
      out << "Ghost::AroundFacesZ";
      break;
    case Ghost::AroundFaces :
      out << "Ghost::AroundFaces";
      break;
    default :
      out << "Unknown ghost type";
      break;
  }
  if (dep.m_gtype != Ghost::None)
    out << ":" << dep.m_num_ghost_cells;

  out << "]";
  return out;
}
  
//______________________________________________________________________
//
std::ostream &
operator <<( std::ostream & out, const Task & task )
{
  task.display(out);
  return out;
}
  
//______________________________________________________________________
//
std::ostream&
operator <<( std::ostream & out, const Task::TaskType & tt )
{
  switch (tt) {
    case Task::Normal :
      out << "Normal";
      break;
    case Task::Reduction :
      out << "Reduction";
      break;
    case Task::InitialSend :
      out << "InitialSend";
      break;
    case Task::Output :
      out << "Output";
      break;
    case Task::OncePerProc :
      out << "OncePerProc";
      break;
    case Task::Spatial :
      out << "Spatial";
      break;
  }
  return out;
}

} // end namespace Uintah

//______________________________________________________________________
//
void
Task::displayAll_DOUT( Uintah::Dout& dbg) const
{
  if( dbg.active() ){
    std::ostringstream message;
    displayAll( message );
    DOUT( dbg, message.str() );
  }
}

//______________________________________________________________________
//
void
Task::displayAll( std::ostream & out ) const
{
  display(out);
  out << '\n';
  for (Task::Dependency* req = m_req_head; req != nullptr; req = req->m_next) {
    out << Parallel::getMPIRank() << "  requires: " << *req << '\n';
  }
  for (Task::Dependency* comp = m_comp_head; comp != nullptr; comp = comp->m_next) {
    out << Parallel::getMPIRank() << "  computes: " << *comp << '\n';
  }
  for (Task::Dependency* mod = m_mod_head; mod != nullptr; mod = mod->m_next) {
    out << Parallel::getMPIRank() << "  modifies: " << *mod << '\n';
  }
}

//______________________________________________________________________
//
void
Task::setMapping( int dwmap[TotalDWs] )
{
  for (int i = 0; i < TotalDWs; i++) {
    this->m_dwmap[i] = dwmap[i];
  }
}

//______________________________________________________________________
//
int
Task::mapDataWarehouse( WhichDW dw ) const
{
  ASSERTRANGE(dw, 0, Task::TotalDWs);
  return m_dwmap[dw];
}

//______________________________________________________________________
//
DataWarehouse *
Task::mapDataWarehouse( WhichDW dw, std::vector<DataWarehouseP> & dws ) const
{
  ASSERTRANGE(dw, 0, Task::TotalDWs);
  if (m_dwmap[dw] == Task::NoDW) {
    return nullptr;
  }
  else {
    ASSERTRANGE(m_dwmap[dw], 0, (int )dws.size());
    return dws[m_dwmap[dw]].get_rep();
  }
}

