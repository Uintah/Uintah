/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Grid.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Parallel/Parallel.h>
#include <set>


using namespace std;
using namespace Uintah;
using namespace SCIRun;

MaterialSubset* Task::globalMatlSubset = 0;

void Task::initialize()
{
  comp_head = comp_tail = 0;
  req_head = req_tail = 0;
  mod_head = mod_tail = 0;
  patch_set = 0;
  matl_set = 0;
  d_usesMPI = false;
  d_usesThreads = false;
  d_usesGPU = false;
  d_subpatchCapable = false;
  d_hasSubScheduler = false;

  for(int i=0;i<TotalDWs;i++) {
    dwmap[i]=Task::InvalidDW;
  }
  sortedOrder=-1;
  d_phase=-1;
  d_comm=-1;
  maxGhostCells=0;
}

Task::ActionBase::~ActionBase()
{
}

Task::ActionGPUBase::~ActionGPUBase()
{
}

Task::~Task()
{
  delete d_action;
  delete d_actionGPU;

  Dependency* dep = req_head;
  while(dep){
    Dependency* next = dep->next;
    delete dep;
    dep=next;
  }

  dep = comp_head;
  while(dep){
    Dependency* next = dep->next;
    delete dep;
    dep=next;
  }

  dep = mod_head;
  while(dep){
    Dependency* next = dep->next;
    delete dep;
    dep=next;
  }
  
  if(matl_set && matl_set->removeReference()) {
    delete matl_set;
  }

  if(patch_set && patch_set->removeReference()) {
    delete patch_set;
  }

  // easier to periodically delete this than to force a call to a cleanup
  // function, and probably not very expensive.
  if (globalMatlSubset && globalMatlSubset->removeReference()) {
    delete globalMatlSubset;
  }

  globalMatlSubset = 0;
}

//__________________________________
void Task::setSets(const PatchSet* ps, const MaterialSet* ms)
{
  ASSERT(patch_set == 0);
  ASSERT(matl_set == 0);
  patch_set=ps;
  if(patch_set) {
    patch_set->addReference();
  }
  matl_set=ms;
  if(matl_set) {
    matl_set->addReference();
  }
}

//__________________________________
const MaterialSubset* Task::getGlobalMatlSubset()
{
  if (globalMatlSubset == 0) {
    globalMatlSubset = scinew MaterialSubset();
    globalMatlSubset->add(-1);
    globalMatlSubset->addReference();
  }
  return globalMatlSubset;
}

void
Task::usesMPI(bool state)
{
  d_usesMPI = state;
}

void
Task::hasSubScheduler(bool state)
{
  d_hasSubScheduler = state;
}

void
Task::usesThreads(bool state)
{
  d_usesThreads = state;
}

void
Task::usesGPU(bool state)
{
  d_usesGPU = state;
}

void
Task::subpatchCapable(bool state)
{
  d_subpatchCapable = state;
}

//__________________________________
void
Task::requires(WhichDW dw, 
               const VarLabel* var,
	        const PatchSubset* patches,
	        PatchDomainSpec patches_dom,
	        int level_offset,
	        const MaterialSubset* matls,
	        MaterialDomainSpec matls_dom,
	        Ghost::GhostType gtype,
	        int numGhostCells,
	        bool oldTG)
{
  if (matls == 0 && var->typeDescription()->isReductionVariable()) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  }
  else if (matls != 0 && matls->size() == 0) {
    return; // no materials, no dependency
  }

  Dependency* dep = scinew Dependency(Requires, this, dw, var, oldTG, patches, matls,
                                      patches_dom, matls_dom,
                                      gtype, numGhostCells, level_offset);

  if (numGhostCells > maxGhostCells) maxGhostCells=numGhostCells;
  if (level_offset > maxLevelOffset) maxLevelOffset=level_offset;

  dep->next=0;
  if(req_tail)
    req_tail->next=dep;
  else
    req_head=dep;
  req_tail=dep;
  if (dw == OldDW)
    d_requiresOldDW.insert(make_pair(var, dep));
  else
    d_requires.insert(make_pair(var, dep));
}

//__________________________________
void
Task::requires(WhichDW dw, 
               const VarLabel* var,
	        const PatchSubset* patches,
	        PatchDomainSpec patches_dom,
	        const MaterialSubset* matls,
	        MaterialDomainSpec matls_dom,
	        Ghost::GhostType gtype,
	        int numGhostCells,
	        bool oldTG)
{
  int offset=0;
  if (patches_dom == CoarseLevel || patches_dom == FineLevel) offset=1;
  requires(dw, var, patches, patches_dom, offset, matls, matls_dom,
           gtype, numGhostCells, oldTG);
}

//__________________________________
void
Task::requires(WhichDW dw, 
              const VarLabel* var,
		const PatchSubset* patches,
		const MaterialSubset* matls,
		Ghost::GhostType gtype,
		int numGhostCells,
		bool oldTG)
{
  requires(dw, var, patches, ThisLevel, matls, NormalDomain,
           gtype, numGhostCells, oldTG);
}

//__________________________________
void
Task::requires(WhichDW dw, 
               const VarLabel* var,
		 Ghost::GhostType gtype,
		 int numGhostCells,
		 bool oldTG)
{
  requires(dw, var, 0, ThisLevel, 0, NormalDomain, gtype, numGhostCells, oldTG);
}

//__________________________________
void
Task::requires(WhichDW dw, 
               const VarLabel* var,
		 const MaterialSubset* matls,
		 Ghost::GhostType gtype,
		 int numGhostCells,
		 bool oldTG)

{
  requires(dw, var, 0, ThisLevel, matls, NormalDomain, gtype, numGhostCells, oldTG);
}

//__________________________________
void
Task::requires(WhichDW dw, 
               const VarLabel* var,
		 const MaterialSubset* matls,
		 MaterialDomainSpec matls_dom,
		 Ghost::GhostType gtype,
		 int numGhostCells,
		 bool oldTG)
{
  requires(dw, var, 0, ThisLevel, matls, matls_dom, gtype, numGhostCells, oldTG);
}

//__________________________________
void
Task::requires(WhichDW dw,
               const VarLabel* var,
               const PatchSubset* patches,
               Ghost::GhostType gtype, 
               int numGhostCells, 
               bool oldTG)
{
  requires(dw, var, patches, ThisLevel, 0, NormalDomain, gtype, numGhostCells, oldTG);
}

//__________________________________
void
Task::requires(WhichDW dw, 
               const VarLabel* var,
               const PatchSubset* patches,
               const MaterialSubset * matls)
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if (vartype == TypeDescription::SoleVariable)
    requires(dw, var, (const Level*)0, matls);
  else if(vartype == TypeDescription::PerPatch )
    requires(dw,var,patches,ThisLevel,matls,NormalDomain,Ghost::None,0);
  else
    SCI_THROW(InternalError("Requires should specify ghost type or level for this variable", __FILE__, __LINE__));
}
//__________________________________
void
Task::requires(WhichDW dw, 
               const VarLabel * var,
               const MaterialSubset * matls,
               bool oldTG)
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if(!(vartype == TypeDescription::PerPatch
       || vartype == TypeDescription::ReductionVariable
       || vartype == TypeDescription::SoleVariable))
    SCI_THROW(InternalError("Requires should specify ghost type for this variable", __FILE__, __LINE__));
    
  if(vartype == TypeDescription::ReductionVariable)
    requires(dw, var, (const Level*)0, matls, NormalDomain, oldTG);
  else if(vartype == TypeDescription::SoleVariable)
    requires(dw, var, (const Level*)0, matls);
  else
    requires(dw, var, 0, ThisLevel, matls, NormalDomain, Ghost::None, 0, oldTG);
}

//__________________________________
void
Task::requires(WhichDW dw, 
               const VarLabel* var,
		 const Level* level,
		 const MaterialSubset * matls,
		 MaterialDomainSpec matls_dom,
		 bool oldTG)
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if(!(vartype == TypeDescription::ReductionVariable ||
       vartype == TypeDescription::SoleVariable))
    SCI_THROW(InternalError("Requires should specify ghost type for this variable", __FILE__, __LINE__));

  if (matls == 0){
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  } else if (matls->size() == 0) {
    return; // no materials, no dependency
  }
  Dependency* dep = scinew Dependency(Requires, this, dw, var, oldTG, level, matls, matls_dom);
  dep->next=0;
  if(req_tail)
    req_tail->next=dep;
  else
    req_head=dep;
  req_tail=dep;
  if (dw == OldDW)
    d_requiresOldDW.insert(make_pair(var, dep));
  else
    d_requires.insert(make_pair(var, dep));
}

//__________________________________
void
Task::computes(const VarLabel * var,
		 const PatchSubset * patches,
		 PatchDomainSpec patches_dom,
		 const MaterialSubset * matls,
		 MaterialDomainSpec matls_dom)
{
  if (var->typeDescription()->isReductionVariable()) {
    if (matls == 0) {
      // default material for a reduction variable is the global material (-1)
      matls = getGlobalMatlSubset();
      matls_dom = OutOfDomain;
    }
    ASSERT(patches == 0);
  }
  
  Dependency* dep = scinew Dependency(Computes, this, NewDW, var, false, patches, matls,
                                      patches_dom, matls_dom);
  dep->next=0;
  if(comp_tail)
    comp_tail->next=dep;
  else
    comp_head=dep;
  comp_tail=dep;

  d_computes.insert(make_pair(var, dep));
}

//__________________________________
void
Task::computes(const VarLabel * var,
		const PatchSubset * patches,
		const MaterialSubset * matls)
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if (vartype == TypeDescription::ReductionVariable ||
      vartype == TypeDescription::SoleVariable)
    computes(var, (const Level*)0, matls);
  else
    computes(var, patches, ThisLevel, matls, NormalDomain);
}

//__________________________________
void
Task::computes(const VarLabel* var, const MaterialSubset* matls)
{
  computes(var, 0, ThisLevel, matls, NormalDomain);
}

//__________________________________
void
Task::computes(const VarLabel* var, const MaterialSubset* matls,
               MaterialDomainSpec matls_dom)
{
  computes(var, 0, ThisLevel, matls, matls_dom);
}

//__________________________________
void
Task::computes(const VarLabel* var, const PatchSubset* patches,
               PatchDomainSpec patches_dom)
{
  computes(var, patches, patches_dom, 0, NormalDomain);
}

//__________________________________
void
Task::computes(const VarLabel* var,
		const Level* level,
		const MaterialSubset * matls,
		MaterialDomainSpec matls_dom)
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if (!(vartype == TypeDescription::ReductionVariable ||
      vartype == TypeDescription::SoleVariable))
    SCI_THROW(InternalError("Computes should only be used for reduction variable", __FILE__, __LINE__));

  if (matls == 0) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  } else if(matls->size() == 0){
    throw InternalError("Computes of an empty material set!", __FILE__, __LINE__);
  }
  
  Dependency* dep = scinew Dependency(Computes, this, NewDW, var, false, level,
                                      matls, matls_dom);
  dep->next=0;
  if(comp_tail)
    comp_tail->next=dep;
  else
    comp_head=dep;
  comp_tail=dep;

  d_computes.insert(make_pair(var, dep));
}

//__________________________________
void
Task::modifies(const VarLabel* var,
		const PatchSubset* patches,
		PatchDomainSpec patches_dom,
		const MaterialSubset* matls,
		MaterialDomainSpec matls_dom,
		bool oldTG)
{
  if (matls == 0 && var->typeDescription()->isReductionVariable()) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
    ASSERT(patches == 0);
  }  

  Dependency* dep = scinew Dependency(Modifies, this, NewDW, var, oldTG, patches, matls,
                                      patches_dom, matls_dom);
  dep->next=0;
  if (mod_tail)
    mod_tail->next=dep;
  else
    mod_head=dep;
  mod_tail=dep;

  d_requires.insert(make_pair(var, dep));
  d_computes.insert(make_pair(var, dep));
  d_modifies.insert(make_pair(var, dep));
}

//__________________________________
void 
Task::modifies(const VarLabel* var,
		 const Level* level,
		 const MaterialSubset* matls,
		 MaterialDomainSpec matls_domain,
		 bool oldTG)
{
  const TypeDescription* vartype = var->typeDescription();
  
  if (matls == 0 && vartype->isReductionVariable()) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_domain = OutOfDomain;
  }  

  if (!vartype->isReductionVariable())
    SCI_THROW(InternalError("modifies with level should only be used for reduction variable", __FILE__, __LINE__));


  Dependency* dep = scinew Dependency(Modifies, this, NewDW, var, oldTG, level, matls,
                                      matls_domain);
  dep->next=0;
  if (mod_tail)
    mod_tail->next=dep;
  else
    mod_head=dep;
  mod_tail=dep;

  d_requires.insert(make_pair(var, dep));
  d_computes.insert(make_pair(var, dep));
  d_modifies.insert(make_pair(var, dep));
}

//__________________________________
void
Task::modifies(const VarLabel* var,
		const PatchSubset* patches,
		const MaterialSubset* matls,
		bool oldTG)
{
  modifies(var, patches, ThisLevel, matls, NormalDomain, oldTG);
}

//__________________________________
void
Task::modifies(const VarLabel* var, bool oldTG)
{
  modifies(var, 0, ThisLevel, 0, NormalDomain, oldTG);
}

//__________________________________
void
Task::modifies(const VarLabel* var, const MaterialSubset* matls, bool oldTG)
{
  modifies(var, 0, ThisLevel, matls, NormalDomain, oldTG);
}

//__________________________________
void
Task::modifies(const VarLabel* var, const MaterialSubset* matls,
               MaterialDomainSpec matls_dom, bool oldTG)
{
  modifies(var, 0, ThisLevel, matls, matls_dom, oldTG);
}

//__________________________________
bool Task::hasComputes(const VarLabel* var, int matlIndex,
                       const Patch* patch) const
{
  return isInDepMap(d_computes, var, matlIndex, patch);
}

//__________________________________
bool Task::hasRequires(const VarLabel * var,
		         int matlIndex,
		         const Patch * patch,
		         IntVector lowOffset,
		         IntVector highOffset,
		         WhichDW dw)const
{
  DepMap depMap = d_requires;
  
  if(dw == OldDW){
    depMap = d_requiresOldDW;
  }
  
  Dependency* dep = isInDepMap(depMap, var, matlIndex, patch);  
  
    
  if (dep) {
    // make sure we are within the allowed ghost cell limit
    IntVector allowableLowOffset, allowableHighOffset;  
    Patch::getGhostOffsets(var->typeDescription()->getType(), dep->gtype,
                           dep->numGhostCells, allowableLowOffset,
                           allowableHighOffset);
    return ((Max(allowableLowOffset, lowOffset) == allowableLowOffset) &&
            (Max(allowableHighOffset, highOffset) == allowableHighOffset));
  }
  return false;
}

//__________________________________
bool Task::hasModifies(const VarLabel* var, int matlIndex,
                       const Patch* patch) const
{
  return isInDepMap(d_modifies, var, matlIndex, patch);
}

//__________________________________
Task::Dependency* Task::isInDepMap(const DepMap& depMap, 
                                   const VarLabel* var,
                                   int matlIndex, 
                                   const Patch* patch) const
{
  DepMap::const_iterator found_iter = depMap.find(var);
  
  // loop over dependency map and search for the right dependency
  
  while (found_iter != depMap.end() && (*found_iter).first->equals(var)) {
  
    Dependency* dep = (*found_iter).second;
    const PatchSubset* patches = dep->patches;
    const MaterialSubset* matls = dep->matls;

    bool hasPatches=false, hasMatls=false;

    if(patches==0) //if patches==0 then the requirement for patches is satisfied
    {
      hasPatches=true;
    }
    else
    {
      if(dep->patches_dom == Task::CoarseLevel)  //check that the level of the patches matches the coarse level
      {
        hasPatches=getLevel(getPatchSet())->getRelativeLevel(-dep->level_offset)==getLevel(patches);
      }
      else if(dep->patches_dom == Task::FineLevel) //check that the level of the patches matches the fine level
      {
        hasPatches=getLevel(getPatchSet())->getRelativeLevel(dep->level_offset)==getLevel(patches);
      }
      else  //check that the patches subset contain the requested patch
      {
        hasPatches=patches->contains(patch);
      }
    }
    
    if (matls == 0) //if matls==0 then the requierment for matls is satisfied
    {
      hasMatls=true;
    }
    else  //check thta the malts subset contains the matlIndex
    {
      hasMatls=matls->contains(matlIndex);
    }
   
    if(hasMatls && hasPatches)  //if this dependency contains both the matls and patches return the dependency
      return dep;

    found_iter++;
  }
  return 0;
}
//__________________________________
//
Task::Dependency::Dependency(DepType deptype, 
                             Task* task, 
                              WhichDW whichdw,
			        const VarLabel* var,
			        bool oldTG,
			        const PatchSubset* patches,
			        const MaterialSubset* matls,
			        PatchDomainSpec patches_dom,
			        MaterialDomainSpec matls_dom,
			        Ghost::GhostType gtype,
			        int numGhostCells,
			        int level_offset)
                             
: deptype(deptype), task(task), var(var), lookInOldTG(oldTG), patches(patches), matls(matls),
  reductionLevel(0), patches_dom(patches_dom), matls_dom(matls_dom),
  gtype(gtype), whichdw(whichdw), numGhostCells(numGhostCells), level_offset(level_offset)
{
  if (var)
    var->addReference();
  req_head=req_tail=comp_head=comp_tail=0;
  if(patches)
    patches->addReference();
  if(matls)
    matls->addReference();
}

//__________________________________
Task::Dependency::Dependency(DepType deptype, 
                             Task* task, 
                             WhichDW whichdw,
			        const VarLabel* var,
			        bool oldTG,
			        const Level* reductionLevel,
			        const MaterialSubset* matls,
			        MaterialDomainSpec matls_dom)

: deptype(deptype), task(task), var(var), lookInOldTG(oldTG), patches(0), matls(matls),
  reductionLevel(reductionLevel), patches_dom(ThisLevel),
  matls_dom(matls_dom), gtype(Ghost::None), whichdw(whichdw), numGhostCells(0), level_offset(0)
{
  if (var)
    var->addReference();
  req_head=req_tail=comp_head=comp_tail=0;
  if(matls)
    matls->addReference();
}

Task::Dependency::~Dependency()
{
  VarLabel::destroy(var); // just remove the ref
  if(patches && patches->removeReference())
    delete patches;
  if(matls && matls->removeReference())
    delete matls;
}

// for xlC:
namespace Uintah {

  /*
template <class T>
constHandle< ComputeSubset<T> > Task::Dependency::
getComputeSubsetUnderDomain(string domString, Task::MaterialDomainSpec dom,
                            const ComputeSubset<T>* subset,
                            const ComputeSubset<T>* domainSubset)
{
  switch(dom){
  case Task::NormalDomain:
  case Task::OtherGridDomain: // use the same patches, we'll figure out where it corresponds on the other grid
    return ComputeSubset<T>::intersection(subset, domainSubset);
  case Task::OutOfDomain:
    return subset;
  case Task::CoarseLevel:
  case Task::FineLevel:      
    return getOtherLevelComputeSubset(dom, subset, domainSubset);
  default:
    SCI_THROW(InternalError(string("Unknown ") + domString + " type "+to_string(static_cast<int>(dom)),
                            __FILE__, __LINE__));
  }
}
*/

//__________________________________
constHandle<PatchSubset>
Task::Dependency::getPatchesUnderDomain(const PatchSubset* domainPatches) const
{
  switch(patches_dom){
  case Task::ThisLevel:
  case Task::OtherGridDomain: // use the same patches, we'll figure out where it corresponds on the other grid
    return PatchSubset::intersection(patches, domainPatches);
  case Task::CoarseLevel:
  case Task::FineLevel:      
    return getOtherLevelPatchSubset(patches_dom, level_offset, patches, domainPatches, numGhostCells);
  default:
    SCI_THROW(InternalError(string("Unknown patch domain ") + " type "+to_string(static_cast<int>(patches_dom)),
                            __FILE__, __LINE__));
  }
}

//__________________________________
constHandle<MaterialSubset>
Task::Dependency::getMaterialsUnderDomain(const MaterialSubset* domainMaterials) const
{
  switch(matls_dom){
  case Task::NormalDomain:
    return MaterialSubset::intersection(matls, domainMaterials);
  case Task::OutOfDomain:
    return matls;
  default:
    SCI_THROW(InternalError(string("Unknown matl domain ") + " type "+to_string(static_cast<int>(matls_dom)),
                            __FILE__, __LINE__));
  }
}

//__________________________________
constHandle< PatchSubset > Task::Dependency::
getOtherLevelPatchSubset(Task::PatchDomainSpec dom, int level_offset,
                         const PatchSubset* subset,
                         const PatchSubset* domainSubset, int ngc)
{
  constHandle<PatchSubset> myLevelSubset =
    PatchSubset::intersection(subset, domainSubset);

  int levelOffset = 0;
  switch(dom){
  case Task::CoarseLevel:
    levelOffset = -level_offset;
    break;
  case Task::FineLevel:
    levelOffset = level_offset;
    break;
  default:
    SCI_THROW(InternalError("Unhandled DomainSpec in Task::Dependency::getOtherLevelComputeSubset",
                            __FILE__, __LINE__));
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

//__________________________________
void
Task::doit(const ProcessorGroup* pg,
	         const PatchSubset* patches,
	         const MaterialSubset* matls,
	         vector<DataWarehouseP>& dws)
{
  DataWarehouse* fromDW = mapDataWarehouse(Task::OldDW, dws);
  DataWarehouse* toDW = mapDataWarehouse(Task::NewDW, dws);
  if(d_action) {
    d_action->doit(pg, patches, matls, fromDW, toDW);
  }
}

void
Task::doitGPU(const ProcessorGroup* pg,
              const PatchSubset* patches,
              const MaterialSubset* matls,
              vector<DataWarehouseP>& dws,
              int device)
{
  DataWarehouse* fromDW = mapDataWarehouse(Task::OldDW, dws);
  DataWarehouse* toDW = mapDataWarehouse(Task::NewDW, dws);
  if(d_actionGPU) {
    d_actionGPU->doitGPU(pg, patches, matls, fromDW, toDW, device);
  }
}

//__________________________________
void
Task::display( ostream & out ) const
{
  out <<  Parallel::getMPIRank()<< " " << getName() << " (" << d_tasktype << "): [";
  out << *patch_set;
  if( d_tasktype ==  Task::Normal && patch_set != NULL){
    out << ", Level " << getLevel(patch_set)->getIndex();
  }
  out << ", ";
  out << *matl_set;
  out << ", DWs: ";
  for(int i=0;i<TotalDWs;i++){
    if(i != 0)
      out << ", ";
    out << dwmap[i];
  }
  out << "]";
}
//__________________________________
namespace Uintah {
  std::ostream &
  operator << ( std::ostream & out, const Uintah::Task::Dependency & dep )
  {
    out << "[";
    out<< left;out.width(20);
    out << *(dep.var) << ", ";
     
    // reduction variable 
    if(dep.var->typeDescription()->isReductionVariable()){
      if(dep.reductionLevel) {
        out << " reduction Level: " << dep.reductionLevel->getIndex();
      } else {
        out << " Global level";
      }
    } else {
      // all other variables:
      if( dep.patches ){
        out << " Level: " << getLevel(dep.patches)->getIndex();
        out << " Patches: ";
        for(int i=0;i<dep.patches->size();i++){
          if(i > 0){
            out << ",";
          }
          out << dep.patches->get(i)->getID();
        }
      } 
      else if(dep.reductionLevel) {
        out << " reduction Level: " << dep.reductionLevel->getIndex();
      } 
      else if(dep.patches_dom){
        switch(dep.patches_dom){
        case Task::CoarseLevel:
          out << "coarseLevel";
          break;
        case Task::FineLevel:
          out << "fineLevel";
          break;
        case Task::OtherGridDomain:
          out << "OtherGridDomain";
          break;  
         case Task::ThisLevel:
          out << "ThisLevel";
          break;
        default:
          break;
        }
      }else{
        out << "all Patches";
      }
    }

    out << ", MI: ";
    if(dep.matls){
      for(int i=0;i<dep.matls->size();i++){
        if(i>0)
          out << ",";
        out << dep.matls->get(i);
      }
    } else {
      out << "none";
    }
    out << ", ";
    switch(dep.whichdw){
    case Task::OldDW:
      out << "OldDW";
      break;
    case Task::NewDW:
      out << "NewDW";
      break;
    case Task::CoarseOldDW:
      out << "CoarseOldDW";
      break;
    case Task::CoarseNewDW:
      out << "CoarseNewDW";
      break;
    case Task::ParentOldDW:
      out << "ParentOldDW";
      break;
    case Task::ParentNewDW:
      out << "ParentNewDW";
      break;
    default:
      out << "Unknown DW!";
      break;
    }
    out << " (mapped to dw index " << dep.task->mapDataWarehouse(dep.whichdw) << ")";
    out << ", ";
    switch(dep.gtype){
    case Ghost::None:
      out << "Ghost::None";
      break;
    case Ghost::AroundNodes:
      out << "Ghost::AroundNodes";
      break;
    case Ghost::AroundCells:
      out << "Ghost::AroundCells";
      break;
    case Ghost::AroundFacesX:
      out << "Ghost::AroundFacesX";
      break;
    case Ghost::AroundFacesY:
      out << "Ghost::AroundFacesY";
      break;
    case Ghost::AroundFacesZ:
      out << "Ghost::AroundFacesZ";
      break;
    case Ghost::AroundFaces:
      out << "Ghost::AroundFaces";
      break;
    default:
      out << "Unknown ghost type";
      break;
    }
    if(dep.gtype != Ghost::None)
      out << ":" << dep.numGhostCells;

    out << "]";
    return out;
  }
  
//__________________________________
  ostream &
  operator << (ostream& out, const Task& task)
  {
    task.display( out );
    return out;
  }
  
//__________________________________
  ostream&
  operator << (ostream &out, const Task::TaskType & tt)
  {
    switch( tt ) {
    case Task::Normal:
      out << "Normal";
      break;
    case Task::Reduction:
      out << "Reduction";
      break;
    case Task::InitialSend:
      out << "InitialSend";
      break;
    case Task::Output:
      out << "Output";
      break;
    case Task::OncePerProc:
      out << "OncePerProc";
      break;
    }
    return out;
  }
} // end namespace Uintah

//__________________________________
void
Task::displayAll(ostream& out) const
{
   display(out);
   out << '\n';
   for(Task::Dependency* req = req_head; req != 0; req = req->next)
      out << Parallel::getMPIRank() << "  requires: " << *req << '\n';
   for(Task::Dependency* comp = comp_head; comp != 0; comp = comp->next)
      out << Parallel::getMPIRank() <<"  computes: " << *comp << '\n';
   for(Task::Dependency* mod = mod_head; mod != 0; mod = mod->next)
      out << Parallel::getMPIRank() <<"  modifies: " << *mod << '\n';
}

//__________________________________
void Task::setMapping(int dwmap[TotalDWs])
{
  for(int i=0;i<TotalDWs;i++)
  {
    this->dwmap[i]=dwmap[i];
  }
}

//__________________________________
int Task::mapDataWarehouse(WhichDW dw) const
{
  ASSERTRANGE(dw, 0, Task::TotalDWs);
  return dwmap[dw];
}

//__________________________________
DataWarehouse* Task::mapDataWarehouse(WhichDW dw, vector<DataWarehouseP>& dws) const
{
  ASSERTRANGE(dw, 0, Task::TotalDWs);
  if(dwmap[dw] == Task::NoDW){
    return 0;
  } else {
    ASSERTRANGE(dwmap[dw], 0, (int)dws.size());
    return dws[dwmap[dw]].get_rep();
  }
}

