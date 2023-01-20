/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

using namespace Uintah;

MaterialSubset* Task::globalMatlSubset = nullptr;

namespace {
  Uintah::MasterLock      deviceNums_mutex{};
  Uintah::MasterLock     cudaStreams_mutex{};
}

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

  // easier to periodically delete this than to force a call to a
  // cleanup function, and probably not very expensive.
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

  m_uses_mpi           = false;
  m_uses_threads       = false;
  m_uses_device        = false;
  m_preload_sim_vars   = false;
  m_uses_kokkos_openmp = false;
  m_uses_kokkos_openmptarget = false;
  m_uses_kokkos_cuda   = false;
  m_subpatch_capable   = false;
  m_has_subscheduler   = false;

  for (int i = 0; i < TotalDWs; i++) {
    m_dwmap[i] = Task::InvalidDW;
  }

  m_sorted_order = -1;
  m_phase        = -1;
  m_comm         = -1;

  // The 0th level has a max ghost cell of zero.  Other levels are left uninitialized.
  m_max_ghost_cells[0] = 0;
  m_max_level_offset   = 0;

  // Assures that CPU tasks will have 1 stream
  if(Uintah::Parallel::usingDevice())
    m_max_streams_per_task = 1;
}

//______________________________________________________________________

//
void
Task::setSets( const PatchSet* ps, const MaterialSet* ms )
{
  // NOTE: the outer [patch/matl]Set checks are related to temporal
  // scheduling, e.g. more then 1 regular task graph
  //
  // This is called from TaskGraph::addTask() in which a single task
  // may be added to more than 1 Normal task graphs. In this case,
  // first time here, m_path/matl_set will be nullptr and subsequent
  // visits will be the same pointer as ps and ms
  // respectively. Without these checks, the refCount gets
  // artificially inflated and ComputeSubsets (Patch/Matl)) are not
  // deleted, resulting in a mem leak. APH, 06/08/17
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
void Task::setExecutionAndMemorySpace( const TaskAssignedExecutionSpace & executionSpaceTypeName
                                     , const TaskAssignedMemorySpace    & memorySpaceTypeName
                                     )
{
  m_execution_space = executionSpaceTypeName;
  m_memory_space = memorySpaceTypeName;
}

//______________________________________________________________________
//
void
Task::usesDevice(bool state, int maxStreamsPerTask /* = -1 */ )
{
  m_uses_device = state;

  if (maxStreamsPerTask == -1) {
    // The default case, get it from a command line argument
    m_max_streams_per_task = Uintah::Parallel::getCudaStreamsPerTask();
  } else {
    // Let the user override it
    m_max_streams_per_task = maxStreamsPerTask;
  }
}

//______________________________________________________________________
//
void
Task::usesSimVarPreloading(bool state)
{
  m_preload_sim_vars = state;
}

//______________________________________________________________________
//
void
Task::usesKokkosOpenMP(bool state)
{
  m_uses_kokkos_openmp = state;
}

//______________________________________________________________________
//
void
Task::usesKokkosOpenMPTarget(bool state)
{
  m_uses_kokkos_openmptarget = state;
}

//______________________________________________________________________
//
void
Task::usesKokkosCuda(bool state)
{
  m_uses_kokkos_cuda = state;
}

//______________________________________________________________________
//
TaskAssignedExecutionSpace
Task::getExecutionSpace() const
{
  return m_execution_space;
}

//______________________________________________________________________
//
TaskAssignedMemorySpace
Task::getMemorySpace() const
{
  return m_memory_space;
}

//______________________________________________________________________
//
void Task::requires(       WhichDW              dw
                   , const VarLabel           * var
                   , const PatchSubset        * patches
                   ,       PatchDomainSpec      patches_dom
                   ,       int                  level_offset
                   , const MaterialSubset     * matls
                   ,       MaterialDomainSpec   matls_dom
                   ,       Ghost::GhostType     gtype
                   ,       int                  numGhostCells
                   ,       bool                 oldTG
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
void Task::requires(       WhichDW            dw
                   , const VarLabel         * var
                   , const PatchSubset      * patches
                   , const MaterialSubset   * matls
                   ,       Ghost::GhostType   gtype
                   ,       int                numGhostCells
                   ,       bool               oldTG
                   )
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
    requires(dw, var, (const Level*) nullptr, matls);
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

  if(vartype == TypeDescription::ReductionVariable) {
    requires(dw, var, (const Level*) nullptr, matls, NormalDomain, oldTG);
  }
  else if(vartype == TypeDescription::SoleVariable) {
    requires(dw, var, (const Level*) nullptr, matls);
  }
  else if(vartype == TypeDescription::PerPatch) {
    requires(dw, var, nullptr, ThisLevel, matls, NormalDomain, Ghost::None, 0, oldTG);
  }
  else {
    SCI_THROW(InternalError("Requires should specify ghost type for this variable", __FILE__, __LINE__));
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

  if (vartype == TypeDescription::ReductionVariable ||
      vartype == TypeDescription::SoleVariable) {

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
  else {
    SCI_THROW(InternalError("Requires should specify ghost type for this variable", __FILE__, __LINE__));
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

  if (vartype == TypeDescription::ReductionVariable ||
      vartype == TypeDescription::SoleVariable) {
    computes(var, (const Level*) nullptr, matls);
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
void Task::computes( const VarLabel           * var
                   , const MaterialSubset     * matls
                   ,       MaterialDomainSpec   matls_dom
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

  if ( vartype == TypeDescription::ReductionVariable ||
       vartype == TypeDescription::SoleVariable) {

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
  else {
    SCI_THROW(InternalError("Computes should only be used for reduction variable", __FILE__, __LINE__));
  }
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
  if (matls == nullptr && var->typeDescription()->isReductionVariable()) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
    ASSERT(patches == nullptr);
  }

  Dependency* dep = scinew Dependency(Modifies, this, NewDW, var, oldTG, patches, matls, patches_dom, matls_dom, gtype, numGhostCells);
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
bool Task::hasDistalRequires() const
{
  for (auto dep = m_req_head; dep != nullptr; dep = dep->m_next) {
    if (dep->m_num_ghost_cells >= MAX_HALO_DEPTH) {
      return true;
    }
  }
  return false;
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
      // check that the level of the patches matches the coarse level
      if (dep->m_patches_dom == Task::CoarseLevel) {
        hasPatches = getLevel(getPatchSet())->getRelativeLevel(-dep->m_level_offset) == getLevel(patches);
      }
      // check that the level of the patches matches the fine level
      else if (dep->m_patches_dom == Task::FineLevel) {
        hasPatches = getLevel(getPatchSet())->getRelativeLevel(dep->m_level_offset) == getLevel(patches);
      }
      // check that the patches subset contain the requested patch
      else {
        hasPatches = patches->contains(patch);
      }
    }
    // if matls == nullptr then the requirement for matls is satisfied
    if (matls == nullptr) {
      hasMatls = true;
    }
    else { // check that the malts subset contains the matlIndex
      hasMatls = matls->contains(matlIndex);
    }

    if (hasMatls && hasPatches) {  // if this dependency contains both
                                   // the matls and patches return the
                                   // dependency
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
  VarLabel::destroy(m_var);  // just remove the ref

  if (m_patches != nullptr && m_patches->removeReference()) {
    delete m_patches;
  }

  if (m_matls != nullptr && m_matls->removeReference()) {
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
  case Task::OtherGridDomain: // use the same patches, we'll figure
                              // out where it corresponds on the other grid
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
  constHandle<PatchSubset> myLevelPatchSubset = PatchSubset::intersection(subset, domainSubset);

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
  for (int p = 0; p < myLevelPatchSubset->size(); p++) {
    const Patch* patch = myLevelPatchSubset->get(p);
    Patch::selectType somePatches;
    patch->getOtherLevelPatches(levelOffset, somePatches, ngc);
    patches.insert(somePatches.begin(), somePatches.end());
  }

  return constHandle<PatchSubset>(scinew PatchSubset(patches.begin(), patches.end()));
}

} // end namespace Uintah

//______________________________________________________________________
//
// TODO: Provide an overloaded legacy CPU/non-portable version that
// doesn't use UintahParams or ExecutionObject
void
Task::doit( const PatchSubset           * patches
          , const MaterialSubset        * matls
          , std::vector<DataWarehouseP> & dws
          , UintahParams                & uintahParams
          )
{
  DataWarehouse* fromDW = mapDataWarehouse(Task::OldDW, dws);
  DataWarehouse* toDW   = mapDataWarehouse(Task::NewDW, dws);

  if (m_action) {
    //m_action->doit(patches, matls, fromDW, toDW, uintahParams, execObj);
    m_action->doit(patches, matls, fromDW, toDW, uintahParams);
  }
}

#if defined(HAVE_GPU)
//______________________________________________________________________
//
#ifdef TASK_MANAGES_EXECSPACE
//_____________________________________________________________________________
//
void
Task::assignDevice(intptr_t dTask, unsigned int device_id)
{
  // m_deviceNum = device_id;

  // As m_deviceNums can be touched by multiple threads a mutext is needed.
  deviceNums_mutex.lock();
  {
    m_deviceNums[dTask].insert( device_id );
  }
  deviceNums_mutex.unlock();
}

//_____________________________________________________________________________
// For tasks where there are multiple devices for the task (i.e. data
// archiver output tasks)
Task::deviceNumSet
Task::getDeviceNums(intptr_t dTask)
{
  // As m_deviceNums can be touched by multiple threads a local copy is needed.
  Task::deviceNumSet dNumSet;

  deviceNums_mutex.lock();
  {
    dNumSet = m_deviceNums[dTask];
  }
  deviceNums_mutex.unlock();

  return dNumSet;
}

#ifdef USE_KOKKOS_INSTANCE
//_____________________________________________________________________________
//
//  Task::ActionNonPortableBase
//_____________________________________________________________________________
//

void
Task::ActionNonPortableBase::
assignDevicesAndInstances(intptr_t dTask)
{
  for (int i = 0; i < this->taskPtr->maxStreamsPerTask(); i++) {
   this->assignDevicesAndInstances(dTask, i);
  }
}

//_____________________________________________________________________________
//
void
Task::ActionNonPortableBase::
assignDevicesAndInstances(intptr_t dTask, unsigned int device_id)
{
  if (this->haveKokkosInstanceForThisTask(dTask, device_id) == false) {
    this->taskPtr->assignDevice(dTask, device_id);

    this->setKokkosInstanceForThisTask(dTask, device_id);
  }
}

//_____________________________________________________________________________
//
bool
Task::ActionNonPortableBase::
haveKokkosInstanceForThisTask(intptr_t dTask, unsigned int device_id) const
{
  bool retVal = false;

  // As m_kokkosInstance can be touched by multiple threads a mutext is needed.
  kokkosInstances_mutex.lock();
  {
    auto iter = m_kokkosInstances.find(dTask); // Instances for this task.

    if(iter != m_kokkosInstances.end())
    {
      kokkosInstanceMap iMap = iter->second;
      kokkosInstanceMapIter it = iMap.find(device_id);

      retVal = (it != iMap.end());
    }
  }
  kokkosInstances_mutex.unlock();

  return retVal;
}

//_____________________________________________________________________________
//
Kokkos::DefaultExecutionSpace
Task::ActionNonPortableBase::
getKokkosInstanceForThisTask(intptr_t dTask, unsigned int device_id) const
{
  // As m_kokkosInstance can be touched by multiple threads a mutext is needed.
  kokkosInstances_mutex.lock();
  {
    auto iter = m_kokkosInstances.find(dTask); // Instances for this task.

    if(iter != m_kokkosInstances.end())
    {
      kokkosInstanceMap iMap = iter->second;
      kokkosInstanceMapIter it = iMap.find(device_id);

      if (it != iMap.end())
      {
        kokkosInstances_mutex.unlock();

        return it->second;
      }
    }
  }

  kokkosInstances_mutex.unlock();

  printf("ERROR! - Task::ActionNonPortableBase::getKokkosInstanceForThisTask() - "
           "This task %s does not have an instance assigned for device %d\n",
           this->taskPtr->getName().c_str(), device_id);
    SCI_THROW(InternalError("Detected Kokkos execution failure on task: " +
                            this->taskPtr->getName(), __FILE__, __LINE__));

  Kokkos::DefaultExecutionSpace instance;

  return instance;
}

//_____________________________________________________________________________
//
void
Task::ActionNonPortableBase::
setKokkosInstanceForThisTask(intptr_t dTask,
                             unsigned int device_id)
{
  // if (instance == nullptr) {
  //   printf("ERROR! - Task::ActionNonPortableBase::setKokkosInstanceForThisTask() - "
  //          "A request was made to assign an instance to a nullptr address "
  //          "for this task %s\n", this->taskPtr->getName().c_str());
  //   SCI_THROW(InternalError("A request was made to assign an instance to a "
  //                           "nullptr address for this task :" +
  //                           this->taskPtr->getName() , __FILE__, __LINE__));
  // } else
  if(this->haveKokkosInstanceForThisTask(dTask, device_id) == true) {
    printf("ERROR! - Task::ActionNonPortableBase::setKokkosInstanceForThisTask() - "
           "This task %s already has an instance assigned for device %d\n",
           this->taskPtr->getName().c_str(), device_id);
    SCI_THROW(InternalError("Detected Kokkos execution failure on task: " +
                            this->taskPtr->getName(), __FILE__, __LINE__));
  } else {
    // printf("Task::ActionNonPortableBase::setKokkosInstanceForThisTask() - "
    //        "This task %s now has an instance assigned for device %d\n",
    //        this->taskPtr->getName().c_str(), device_id);
    // As m_kokkosInstances can be touched by multiple threads a
    // mutext is needed.
    kokkosInstances_mutex.lock();
    {
      Kokkos::DefaultExecutionSpace instance;
      m_kokkosInstances[dTask][device_id] = instance;
    }
    kokkosInstances_mutex.unlock();
  }
}

//_____________________________________________________________________________
//
void
Task::ActionNonPortableBase::
clearKokkosInstancesForThisTask(intptr_t dTask)
{
  // As m_kokkosInstances can be touched by multiple threads a mutext is needed.
  kokkosInstances_mutex.lock();
  {
    if(m_kokkosInstances.find(dTask) != m_kokkosInstances.end())
    {
      m_kokkosInstances[dTask].clear();
      m_kokkosInstances.erase(dTask);
    }
  }
  kokkosInstances_mutex.unlock();

  // printf("Task::ActionNonPortableBase::clearKokkosInstancesForThisTask() - "
  //     "Clearing instances for task %s\n",
  //     this->taskPtr->getName().c_str());
}

//_____________________________________________________________________________
//
bool
Task::ActionNonPortableBase::
checkKokkosInstanceDoneForThisTask( intptr_t dTask, unsigned int device_id ) const
{
  // ARS - FIXME
  if (device_id != 0) {
   printf("Error, Task::checkKokkosInstanceDoneForThisTask is %u\n", device_id);
   exit(-1);
  }

  // Base call is commented out
  // OnDemandDataWarehouse::uintahSetCudaDevice(device_id);

  Kokkos::DefaultExecutionSpace instance =
    this->getKokkosInstanceForThisTask(dTask, device_id);

  instance.fence();

  return true;
}

//_____________________________________________________________________________
//
bool
Task::ActionNonPortableBase::
checkAllKokkosInstancesDoneForThisTask(intptr_t dTask) const
{
  // A task can have multiple instances (such as an output task
  // pulling from multiple GPUs).  Check all instacnes to see if they
  // are done.  If any one instance isn't done, return false.  If
  // nothing returned false, then they all must be good to go.
  bool retVal = true;

  // As m_kokkosInstances can be touched by multiple threads get a local
  // copy so not to lock everything.
  kokkosInstanceMap kokkosInstances;

  kokkosInstances_mutex.lock();
  {
    auto iter = m_kokkosInstances.find(dTask);
    if(iter != m_kokkosInstances.end()) {
      kokkosInstances = iter->second;
    } else {
      kokkosInstances_mutex.unlock();

      return retVal;
    }
  }
  kokkosInstances_mutex.unlock();

  for (auto & it : kokkosInstances)
  {
    retVal = this->checkKokkosInstanceDoneForThisTask(dTask, it.first);
    if (retVal == false)
      break;
  }

  return retVal;
}

//_____________________________________________________________________________
//
void
Task::ActionNonPortableBase::
doKokkosDeepCopy( intptr_t dTask, unsigned int deviceNum,
                  void* dst, void* src,
                  size_t count, GPUMemcpyKind kind)
{
  Kokkos::DefaultExecutionSpace instance =
    this->getKokkosInstanceForThisTask(dTask, deviceNum);

  char * srcPtr = static_cast<char *>(src);
  char * dstPtr = static_cast<char *>(dst);

  if(kind == GPUMemcpyHostToDevice)
  {
    // Create an unmanage Kokkos view from the raw pointers.
    Kokkos::View<char*, Kokkos::HostSpace>               hostView( srcPtr, count);
    Kokkos::View<char*, Kokkos::DefaultExecutionSpace> deviceView( dstPtr, count);
    // Deep copy the host view to the device view.
    Kokkos::deep_copy(instance, deviceView, hostView);
  }
  else if(kind == GPUMemcpyDeviceToHost)
  {
    // Create an unmanage Kokkos view from the raw pointers.
    Kokkos::View<char*, Kokkos::HostSpace>               hostView( dstPtr, count);
    Kokkos::View<char*, Kokkos::DefaultExecutionSpace> deviceView( srcPtr, count);
    // Deep copy the device view to the host view.
    Kokkos::deep_copy(instance, hostView, deviceView);
  }
}

//_____________________________________________________________________________
//
void
Task::ActionNonPortableBase::
doKokkosMemcpyPeerAsync( intptr_t dTask,
                       unsigned int deviceNum,
                             void* dst, int  dstDevice,
                       const void* src, int  srcDevice,
                       size_t count )
{
  Kokkos::DefaultExecutionSpace instance =
    this->getKokkosInstanceForThisTask(dTask, deviceNum);
  // ARS - FIXME
  // CUDA_RT_SAFE_CALL(cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice,
  //                                    count, *stream));
}

//_____________________________________________________________________________
//
void
Task::ActionNonPortableBase::
copyGpuGhostCellsToGpuVars(intptr_t dTask,
                           unsigned int deviceNum,
                           GPUDataWarehouse *taskgpudw)
{
  Kokkos::DefaultExecutionSpace instance =
    this->getKokkosInstanceForThisTask(dTask, deviceNum);

  taskgpudw->copyGpuGhostCellsToGpuVarsInvoker<Kokkos::DefaultExecutionSpace>(instance);
}

//_____________________________________________________________________________
//
void
Task::ActionNonPortableBase::
syncTaskGpuDW(intptr_t dTask,
              unsigned int deviceNum,
              GPUDataWarehouse *taskgpudw)
{
  Kokkos::DefaultExecutionSpace instance =
    this->getKokkosInstanceForThisTask(dTask, deviceNum);

  taskgpudw->syncto_device<Kokkos::DefaultExecutionSpace>(instance);
}

//_____________________________________________________________________________
//
//  Task instance pass through methods to the action
//_____________________________________________________________________________
//
void
Task::assignDevicesAndInstances(intptr_t dTask)
{
  if (m_action)
    m_action->assignDevicesAndInstances(dTask);
}

//_____________________________________________________________________________
//
void
Task::assignDevicesAndInstances(intptr_t dTask, unsigned int device_id)
{
  if (m_action)
    m_action->assignDevicesAndInstances(dTask, device_id);
}

//_____________________________________________________________________________
//
void Task::clearKokkosInstancesForThisTask(intptr_t dTask)
{
  if (m_action)
    m_action->clearKokkosInstancesForThisTask(dTask);
}

//_____________________________________________________________________________
//
bool Task::checkAllKokkosInstancesDoneForThisTask(intptr_t dTask) const
{
  if (m_action)
    return m_action->checkAllKokkosInstancesDoneForThisTask(dTask);
  else
    return true;
}

//_____________________________________________________________________________
//
void Task::doKokkosDeepCopy( intptr_t dTask, unsigned int deviceNum,
                             void* dst, void* src,
                             size_t count, cudaMemcpyKind kind)
{
  if (m_action)
    m_action->doKokkosDeepCopy(dTask, deviceNum, dst,src, count,
                               Uintah::Task::GPUMemcpyKind(kind));
}

//_____________________________________________________________________________
//
void Task::doKokkosMemcpyPeerAsync( intptr_t dTask, unsigned int deviceNum,
                                            void* dst, int dstDevice,
                                      const void* src, int srcDevice,
                                      size_t count )
{
  if (m_action) {
    m_action->doKokkosMemcpyPeerAsync(dTask, deviceNum,
                                      dst, dstDevice, src, srcDevice, count);
  }
}

//_____________________________________________________________________________
//
void
Task::copyGpuGhostCellsToGpuVars(intptr_t dTask, unsigned int deviceNum,
                                 GPUDataWarehouse *taskgpudw)
{
  if (m_action) {
    m_action->copyGpuGhostCellsToGpuVars(dTask, deviceNum, taskgpudw);
  }
}

//_____________________________________________________________________________
//
void
Task::syncTaskGpuDW(intptr_t dTask, unsigned int deviceNum,
                    GPUDataWarehouse *taskgpudw)
{
  if (m_action) {
    m_action->syncTaskGpuDW(dTask, deviceNum, taskgpudw);
  }
}
#else  // #ifdef TASK_MANAGES_EXECSPACE - STREAMS
//_____________________________________________________________________________
//
void
Task::assignDevicesAndStreams(intptr_t dTask)
{
  for (int i = 0; i < maxStreamsPerTask(); i++) {
    assignDevicesAndStreams(dTask, i);

    // if (this->haveCudaStreamForThisTask(dTask, i) == false) {
    //   this->assignDevice(dTask, 0);
    //   // this->assignDevice(i);
    //   this->setCudaStreamForThisTask(dTask, i);
    // }
  }
}

//_____________________________________________________________________________
//
void
Task::assignDevicesAndStreams(intptr_t dTask, unsigned int device_id)
{
  if (this->haveCudaStreamForThisTask(dTask, device_id) == false) {
    this->assignDevice(dTask, device_id);
    this->setCudaStreamForThisTask(dTask, device_id);
  }
}

//_____________________________________________________________________________
//
cudaStream_t*
Task::getCudaStreamForThisTask(intptr_t dTask, unsigned int device_id) const
{
  // As m_cudaStreams can be touched by multiple threads a mutext is needed.
  cudaStreams_mutex.lock();
  {
    auto iter = m_cudaStreams.find(dTask); // Streams for this task.

    if( iter != m_cudaStreams.end())
    {
      cudaStreamMap streamMap = iter->second;
      cudaStreamMapIter it = streamMap.find(device_id);

      if (it != streamMap.end())
      {
        cudaStreams_mutex.unlock();

        return it->second;
      }
    }
  }
  cudaStreams_mutex.unlock();

  printf("ERROR! - Task::getCudaStreamForThisTask() - "
         "This task %s does not have a stream assigned for device %d\n",
         getName().c_str(), device_id);
  SCI_THROW(InternalError("Detected CUDA kernel execution failure on task: " +
                          getName(), __FILE__, __LINE__));

  return nullptr;
}

//_____________________________________________________________________________
//
bool
Task::haveCudaStreamForThisTask(intptr_t dTask, unsigned int device_id) const
{
  bool retVal = false;

  // As m_cudaStreams can be touched by multiple threads a mutext is needed.
  cudaStreams_mutex.lock();
  {
    auto iter = m_cudaStreams.find(dTask); // Streams for this task.

    if(iter != m_cudaStreams.end())
    {
      cudaStreamMap streamMap = iter->second;
      cudaStreamMapIter it = streamMap.find(device_id);

      retVal = (it != streamMap.end());
    }
  }
  cudaStreams_mutex.unlock();

  return retVal;
}

//_____________________________________________________________________________
//
void
Task::setCudaStreamForThisTask( intptr_t dTask
                              , unsigned int   device_id
                              )
{
  // Comment out as the stream at this point should never be a nullptr.
  // if (stream == nullptr) {
  //   printf("ERROR! - Task::setCudaStreamForThisTask() - "
  //          "A request was made to assign a stream to a nullptr address "
  //          "for this task %s\n", getName().c_str());
  //   SCI_THROW(InternalError("A request was made to assign a stream to a "
  //                           "nullptr address for this task :" +
  //                           getName() , __FILE__, __LINE__));
  // } else
  if(haveCudaStreamForThisTask(dTask, device_id)) {
    printf("ERROR! - Task::setCudaStreamForThisTask() - "
           "This task %s already has a stream assigned for device %d\n",
           getName().c_str(), device_id);
    SCI_THROW(InternalError("Detected CUDA kernel execution failure on task: " +
                            getName(), __FILE__, __LINE__));
  } else {
    // printf("Task::setCudaStreamForThisTask() - "
    //        "This task %s now has a stream assigned for device %d\n",
    //        getName().c_str(), device_id);
    // As m_cudaStreams can be touched by multiple threads a mutext is needed.
    cudaStreams_mutex.lock();
    {
      cudaStream_t* stream = GPUMemoryPool::getCudaStreamFromPool(this, device_id);
      m_cudaStreams[dTask][device_id] = stream;
    }
    cudaStreams_mutex.unlock();
  }
}

//_____________________________________________________________________________
//
void
Task::reclaimCudaStreamsIntoPool(intptr_t dTask)
{
  // Once streams are reclaimed, clearCudaStreamsForThisTask is called.
  GPUMemoryPool::reclaimCudaStreamsIntoPool(dTask, this);
}

//_____________________________________________________________________________
//
void
Task::clearCudaStreamsForThisTask(intptr_t dTask)
{
  // As m_cudaStreams can be touched by multiple threads a mutext is needed.
  cudaStreams_mutex.lock();
  {
    if(m_cudaStreams.find(dTask) != m_cudaStreams.end())
    {
      m_cudaStreams[dTask].clear();
      m_cudaStreams.erase(dTask);
    }
  }
  cudaStreams_mutex.unlock();
}

//_____________________________________________________________________________
//
bool
Task::checkCudaStreamDoneForThisTask( intptr_t dTask
                                    , unsigned int device_id) const
{
  // sets the CUDA context, for the call to cudaEventQuery()

  if (device_id != 0) {
   printf("Error, Task::checkCudaStreamDoneForThisTask is %u\n", device_id);
   exit(-1);
  }

  // Base call is commented out
  // OnDemandDataWarehouse::uintahSetCudaDevice(device_id);

  cudaStream_t* stream = getCudaStreamForThisTask(dTask, device_id);

  cudaError_t retVal = cudaStreamQuery(*stream);
  if (retVal == cudaSuccess) {
    return true;
  }
  else if (retVal == cudaErrorNotReady ) {
    return false;
  }
  else if (retVal ==  cudaErrorLaunchFailure) {
    printf("ERROR! - Task::checkCudaStreamDoneForThisTask(%d) - "
           "CUDA kernel execution failure on Task: %s\n",
           device_id, getName().c_str());
    SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: " +
                            getName() , __FILE__, __LINE__));
    return false;
  } else { //other error
    printf("\nA CUDA error occurred with error code %d.\n\n"
           "Waiting for 60 seconds\n", retVal);

    int sleepTime = 60;

    struct timespec ts;
    ts.tv_sec = (int) sleepTime;
    ts.tv_nsec = (int)(1.e9 * (sleepTime - ts.tv_sec));

    nanosleep(&ts, &ts);

    CUDA_RT_SAFE_CALL (retVal);
    return false;
  }
}

//_____________________________________________________________________________
//
bool
Task::checkAllCudaStreamsDoneForThisTask(intptr_t dTask) const
{
  // A task can have multiple streams (such as an output task pulling
  // from multiple GPUs).  Check all streams to see if they are done.
  // If any one stream isn't done, return false.  If nothing returned
  // false, then they all must be good to go.
  bool retVal = true;

  // As m_cudaStreams can be touched by multiple threads get a local
  // copy so not to lock everything.
  cudaStreamMap cudaStreams;

  cudaStreams_mutex.lock();
  {
    auto iter = m_cudaStreams.find(dTask); // Streams for this task.

    if(iter != m_cudaStreams.end()) {
      cudaStreams = iter->second;
    } else {
      cudaStreams_mutex.unlock();

      return retVal;
    }
  }
  cudaStreams_mutex.unlock();

  for (auto & it : cudaStreams)
  {
    retVal = checkCudaStreamDoneForThisTask(dTask, it.first);
    if (retVal == false)
      break;
  }

  return retVal;
}


//_____________________________________________________________________________
//
void
Task::doCudaMemcpyAsync( intptr_t dTask, unsigned int deviceNum,
                         void* dst, void* src,
                         size_t count, cudaMemcpyKind kind)
{
  cudaStream_t* stream = getCudaStreamForThisTask(dTask, deviceNum);

#ifdef USE_KOKKOS_DEEPCOPY
  Kokkos::DefaultExecutionSpace instance(*stream);

  char * srcPtr = static_cast<char *>(src);
  char * dstPtr = static_cast<char *>(dst);

  if(kind == GPUMemcpyHostToDevice)
  {
    Kokkos::View<char*, Kokkos::HostSpace>               hostView( srcPtr, count);
    Kokkos::View<char*, Kokkos::DefaultExecutionSpace> deviceView( dstPtr, count);
    // Deep copy the host view to the device view.
    Kokkos::deep_copy(instance, deviceView, hostView);
  }
  else if(kind == GPUMemcpyDeviceToHost)
  {
    Kokkos::View<char*, Kokkos::HostSpace>               hostView( dstPtr, count);
    Kokkos::View<char*, Kokkos::DefaultExecutionSpace> deviceView( srcPtr, count);
    // Deep copy the device view to the host view.
    Kokkos::deep_copy(instance, hostView, deviceView);
  }
#else
  CUDA_RT_SAFE_CALL(cudaMemcpyAsync(dst, src, count, kind, *stream));
#endif
}

//_____________________________________________________________________________
//
void
Task::doCudaMemcpyPeerAsync( intptr_t dTask, unsigned int deviceNum,
                                   void* dst, int  dstDevice,
                             const void* src, int  srcDevice,
                             size_t count )
{
  cudaStream_t* stream = getCudaStreamForThisTask(dTask, deviceNum);

#ifdef USE_KOKKOS_DEEPCOPY
  printf("ERROR! - Task::doCudaMemcpyPeerAsync() - NOT DEFINED"
           "This task %s does not have an stream assigned for device %d\n",
           this->getName().c_str(), deviceNum);
  SCI_THROW(InternalError("Detected Kokkos execution failure on task: " +
                          this->getName(), __FILE__, __LINE__));
#else
  CUDA_RT_SAFE_CALL(cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice,
                                        count, *stream));
#endif
}

//_____________________________________________________________________________
//
void
Task::copyGpuGhostCellsToGpuVars(intptr_t dTask, unsigned int deviceNum,
                                 GPUDataWarehouse *taskgpudw)
{
  cudaStream_t* stream = getCudaStreamForThisTask(dTask, deviceNum);

  taskgpudw->copyGpuGhostCellsToGpuVarsInvoker(stream);

  // Kokkos::DefaultExecutionSpace instanceObject(*stream);
  // taskgpudw->copyGpuGhostCellsToGpuVarsInvoker(instanceObject);
}

//_____________________________________________________________________________
//
void
Task::syncTaskGpuDW(intptr_t dTask, unsigned int deviceNum,
                    GPUDataWarehouse *taskgpudw)
{
  cudaStream_t* stream = getCudaStreamForThisTask(dTask, deviceNum);

  taskgpudw->syncto_device(stream);
}
#endif // #ifdef USE_KOKKOS_INSTANCE - STREAMS
#endif // #ifdef TASK_MANAGES_EXECSPACE
#endif // #if defined(HAVE_GPU)

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
    case Task::Hypre :
      out << "Hypre";
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
