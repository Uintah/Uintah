
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/FancyAssert.h>
#include <iostream>
#include <set>

using namespace Uintah;
using namespace SCIRun;

MaterialSubset* Task::globalMatlSubset = 0;

void Task::initialize()
{
  comp_head=comp_tail=0;
  req_head=req_tail=0;
  mod_head=mod_tail=0;
  patch_set=0;
  matl_set=0;
  d_usesThreads = false;
  d_usesMPI = false;
  d_subpatchCapable = false;
  d_hasSubScheduler = false;
  for(int i=0;i<TotalDWs;i++)
    dwmap[i]=Task::InvalidDW;
  sortedOrder=-1;
}

Task::ActionBase::~ActionBase()
{
}

Task::~Task()
{
  delete d_action;
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
  
  if(matl_set && matl_set->removeReference())
    delete matl_set;

  if(patch_set && patch_set->removeReference())
    delete patch_set;

  // easier to periodically delete this than to force a call to a cleanup
  // function, and probably not very expensive.
  if (globalMatlSubset && globalMatlSubset->removeReference())
    delete globalMatlSubset;
  globalMatlSubset = 0;
}

void Task::setSets(const PatchSet* ps, const MaterialSet* ms)
{
  ASSERT(patch_set == 0);
  ASSERT(matl_set == 0);
  patch_set=ps;
  if(patch_set)
    patch_set->addReference();
  matl_set=ms;
  if(matl_set)
    matl_set->addReference();
}

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
Task::subpatchCapable(bool state)
{
  d_subpatchCapable = state;
}

void
Task::requires(WhichDW dw, const VarLabel* var,
	       const PatchSubset* patches, DomainSpec patches_dom,
	       const MaterialSubset* matls, DomainSpec matls_dom,
	       Ghost::GhostType gtype, int numGhostCells)
{
  if (matls == 0 && var->typeDescription()->isReductionVariable()) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  }
  else if (matls != 0 && matls->size() == 0) {
    return; // no materials, no dependency
  }
  Dependency* dep = scinew Dependency(Requires, this, dw, var, patches, matls,
				      patches_dom, matls_dom,
				      gtype, numGhostCells);
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

void
Task::requires(WhichDW dw, const VarLabel* var,
	       const PatchSubset* patches, const MaterialSubset* matls,
	       Ghost::GhostType gtype, int numGhostCells)
{
  requires(dw, var, patches, NormalDomain, matls, NormalDomain,
	   gtype, numGhostCells);
}


void
Task::requires(WhichDW dw, const VarLabel* var,
	       Ghost::GhostType gtype, int numGhostCells)
{
  requires(dw, var, 0, NormalDomain, 0, NormalDomain, gtype, numGhostCells);
}

void
Task::requires(WhichDW dw, const VarLabel* var,
	       const MaterialSubset* matls,
	       Ghost::GhostType gtype, int numGhostCells)
{
  requires(dw, var, 0, NormalDomain, matls, NormalDomain, gtype, numGhostCells);
}

void
Task::requires(WhichDW dw, const VarLabel* var,
	       const MaterialSubset* matls, DomainSpec matls_dom,
	       Ghost::GhostType gtype, int numGhostCells)
{
  requires(dw, var, 0, NormalDomain, matls, matls_dom, gtype, numGhostCells);
}

void
Task::requires(WhichDW dw, const VarLabel* var,
	       const PatchSubset* patches,
	       Ghost::GhostType gtype, int numGhostCells)
{
  requires(dw, var, patches, NormalDomain, 0, NormalDomain, gtype, numGhostCells);
}

void
Task::requires(WhichDW dw, const VarLabel* var, const PatchSubset* patches,
	       const MaterialSubset* matls)
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if(!vartype == TypeDescription::PerPatch)
    SCI_THROW(InternalError("Requires should specify ghost type or level for this variable"));
  requires(dw, var, patches, NormalDomain, matls, NormalDomain, Ghost::None, 0);
}

void
Task::requires(WhichDW dw, const VarLabel* var, const MaterialSubset* matls)
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if(!(vartype == TypeDescription::PerPatch
       || vartype == TypeDescription::ReductionVariable))
    SCI_THROW(InternalError("Requires should specify ghost type for this variable"));
  if(vartype == TypeDescription::ReductionVariable)
    requires(dw, var, (const Level*)0, matls);
  else
    requires(dw, var, 0, NormalDomain, matls, NormalDomain, Ghost::None, 0);
}

void
Task::requires(WhichDW dw, const VarLabel* var, const Level* level,
	       const MaterialSubset* matls, DomainSpec matls_dom)
{
  TypeDescription::Type vartype = var->typeDescription()->getType();
  if(vartype != TypeDescription::ReductionVariable)
    SCI_THROW(InternalError("Requires should specify ghost type for this variable"));

  if (matls == 0){
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  } else if (matls->size() == 0) {
    return; // no materials, no dependency
  }
  Dependency* dep = scinew Dependency(Requires, this, dw, var, level, matls, matls_dom);
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

void
Task::computes(const VarLabel* var,
	       const PatchSubset* patches, DomainSpec patches_dom,
	       const MaterialSubset* matls, DomainSpec matls_dom)
{
  if (var->typeDescription()->isReductionVariable()) {
    if (matls == 0) {
      // default material for a reduction variable is the global material (-1)
      matls = getGlobalMatlSubset();
      matls_dom = OutOfDomain;
    }
    ASSERT(patches == 0);
  }
  
  Dependency* dep = scinew Dependency(Computes, this, NewDW, var, patches, matls,
				      patches_dom, matls_dom);
  dep->next=0;
  if(comp_tail)
    comp_tail->next=dep;
  else
    comp_head=dep;
  comp_tail=dep;

  d_computes.insert(make_pair(var, dep));
}

void
Task::computes(const VarLabel* var, const PatchSubset* patches,
               const MaterialSubset* matls)
{
  if (var->typeDescription()->isReductionVariable()) 
    computes(var, (const Level*)0, matls);
  else
    computes(var, patches, NormalDomain, matls, NormalDomain);
}

void
Task::computes(const VarLabel* var, const MaterialSubset* matls)
{
  computes(var, 0, NormalDomain, matls, NormalDomain);
}

void
Task::computes(const VarLabel* var, const MaterialSubset* matls,
	       DomainSpec matls_dom)
{
  computes(var, 0, NormalDomain, matls, matls_dom);
}

void
Task::computes(const VarLabel* var, const PatchSubset* patches,
	       DomainSpec patches_dom)
{
  computes(var, patches, patches_dom, 0, NormalDomain);
}

void
Task::computes(const VarLabel* var, const Level* level,
	       const MaterialSubset* matls, DomainSpec matls_dom)
{
  if (!var->typeDescription()->isReductionVariable()) 
    SCI_THROW(InternalError("Computes should only be used for reduction variable"));

  if (matls == 0) {
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
  } else if(matls->size() == 0){
    throw InternalError("Computes of an empty material set!");
  }
  
  Dependency* dep = scinew Dependency(Computes, this, NewDW, var, level,
				      matls, matls_dom);
  dep->next=0;
  if(comp_tail)
    comp_tail->next=dep;
  else
    comp_head=dep;
  comp_tail=dep;

  d_computes.insert(make_pair(var, dep));
}

void
Task::modifies(const VarLabel* var,
	       const PatchSubset* patches, DomainSpec patches_dom,
	       const MaterialSubset* matls, DomainSpec matls_dom)
{
  if (matls == 0 && var->typeDescription()->isReductionVariable()) {
    // in order to implement modifies for reduction variables, the
    // TaskGraph::setupTaskConnections would have to be rewritten for
    // one thing.
    SCI_THROW(InternalError("Modifies not implemented for reduction variables."));
    /*
    // default material for a reduction variable is the global material (-1)
    matls = getGlobalMatlSubset();
    matls_dom = OutOfDomain;
    */
  }  

  Dependency* dep = scinew Dependency(Modifies, this, NewDW, var, patches, matls,
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

void
Task::modifies(const VarLabel* var, const PatchSubset* patches,
               const MaterialSubset* matls)
{
  modifies(var, patches, NormalDomain, matls, NormalDomain);
}

void
Task::modifies(const VarLabel* var, const MaterialSubset* matls)
{
  modifies(var, 0, NormalDomain, matls, NormalDomain);
}

void
Task::modifies(const VarLabel* var, const MaterialSubset* matls,
	       DomainSpec matls_dom)
{
  modifies(var, 0, NormalDomain, matls, matls_dom);
}

bool Task::hasComputes(const VarLabel* var, int matlIndex,
		       const Patch* patch) const
{
  return isInDepMap(d_computes, var, matlIndex, patch);
}

bool Task::hasRequires(const VarLabel* var, int matlIndex,
		       const Patch* patch, IntVector lowOffset,
		       IntVector highOffset, WhichDW dw) const
{
  Dependency* dep =
    isInDepMap((dw == OldDW) ? d_requiresOldDW : d_requires,
	       var, matlIndex, patch);
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

bool Task::hasModifies(const VarLabel* var, int matlIndex,
		       const Patch* patch) const
{
  return isInDepMap(d_modifies, var, matlIndex, patch);
}

Task::Dependency* Task::isInDepMap(const DepMap& depMap, const VarLabel* var,
		      int matlIndex, const Patch* patch) const
{
  DepMap::const_iterator found_iter = depMap.find(var);
  while (found_iter != depMap.end() &&
	 (*found_iter).first->equals(var)) {
    Dependency* dep = (*found_iter).second;
    const PatchSubset* patches = dep->patches;
    const MaterialSubset* matls = dep->matls;
    if (patches == 0) {
      if (!(var->typeDescription() &&
	    var->typeDescription()->isReductionVariable())) {
	patches = getPatchSet() ? getPatchSet()->getUnion() : 0;
      }
    }
    if (matls == 0)
      matls = getMaterialSet() ? getMaterialSet()->getUnion() : 0;
    if (patches == 0 && matls == 0)
      return dep; // assume it is for any matl or patch
    else if (patches == 0) {
      // assume it is for any patch
      if (matls->contains(matlIndex))
	return dep; 
    }
    else if (matls == 0) {
      // assume it is for any matl
      if (patches->contains(patch))
	return dep; 
    }
    else if (patches->contains(patch) && matls->contains(matlIndex))
      return dep;
    found_iter++;
  }
  return 0;
}

Task::Dependency::Dependency(DepType deptype, Task* task, WhichDW whichdw,
			     const VarLabel* var, const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DomainSpec patches_dom,
			     DomainSpec matls_dom,
			     Ghost::GhostType gtype,
			     int numGhostCells)
: deptype(deptype), task(task), var(var), patches(patches), matls(matls),
  reductionLevel(0), patches_dom(patches_dom), matls_dom(matls_dom),
  gtype(gtype), whichdw(whichdw), numGhostCells(numGhostCells)
{
  if (var)
    var->addReference();
  req_head=req_tail=comp_head=comp_tail=0;
  if(patches)
    patches->addReference();
  if(matls)
    matls->addReference();
}

Task::Dependency::Dependency(DepType deptype, Task* task, WhichDW whichdw,
			     const VarLabel* var, const Level* reductionLevel,
			     const MaterialSubset* matls,
			     DomainSpec matls_dom)
: deptype(deptype), task(task), var(var), patches(0), matls(matls),
  reductionLevel(reductionLevel), patches_dom(NormalDomain),
  matls_dom(matls_dom), gtype(Ghost::None), whichdw(whichdw), numGhostCells(0)
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

// MaterialSubset specialization
constHandle< MaterialSubset > Task::Dependency::
getOtherLevelComputeSubset(Task::DomainSpec,
			   const MaterialSubset*,
			   const MaterialSubset*)
{
  // PatchSubset and MaterialSubset specializations are in Task.cc
  SCI_THROW(InternalError("The DomainSpec for a Task::Dependency's MaterialSubset cannot be coarseLevel or FineLevel"));
}

template <class T>
constHandle< ComputeSubset<T> > Task::Dependency::
getComputeSubsetUnderDomain(string domString, Task::DomainSpec dom,
			    const ComputeSubset<T>* subset,
			    const ComputeSubset<T>* domainSubset)
{
  switch(dom){
  case Task::NormalDomain:
    return ComputeSubset<T>::intersection(subset, domainSubset);
  case Task::OutOfDomain:
    return subset;
  case Task::CoarseLevel:
  case Task::FineLevel:      
    return getOtherLevelComputeSubset(dom, subset, domainSubset);
  default:
    SCI_THROW(InternalError(string("Unknown ") + domString + " type "+to_string(static_cast<int>(dom))));
  }
}

constHandle<PatchSubset>
Task::Dependency::getPatchesUnderDomain(const PatchSubset* domainPatches) const
{
  return getComputeSubsetUnderDomain("patches_dom", patches_dom, patches,
				     domainPatches);
}
      
constHandle<MaterialSubset>
Task::Dependency::getMaterialsUnderDomain(const MaterialSubset* domainMaterials) const
{
  return getComputeSubsetUnderDomain("matls_dom", matls_dom, matls,
				     domainMaterials);
}


// PatcheSubset specialization
constHandle< PatchSubset > Task::Dependency::
getOtherLevelComputeSubset(Task::DomainSpec dom,
			   const PatchSubset* subset,
			   const PatchSubset* domainSubset)
{
  constHandle<PatchSubset> myLevelSubset =
    PatchSubset::intersection(subset, domainSubset);

  int levelOffset = 0;
  switch(dom){
  case Task::CoarseLevel:
    levelOffset = -1;
    break;
  case Task::FineLevel:
    levelOffset = 1;
    break;
  default:
    SCI_THROW(InternalError("Unhandled DomainSpec in Task::Dependency::getOtherLevelComputeSubset"));
  }

  std::set<const Patch*, Patch::Compare> patches;
  for (int p = 0; p < myLevelSubset->size(); p++) {
    const Patch* patch = myLevelSubset->get(p);
    Patch::selectType somePatches;
    patch->getOtherLevelPatches(levelOffset, somePatches);
    patches.insert(somePatches.begin(), somePatches.end());
  }

  return constHandle<PatchSubset>(scinew
				  PatchSubset(patches.begin(), patches.end()));
}

} // end namespace Uintah

void
Task::doit(const ProcessorGroup* pc, const PatchSubset* patches,
           const MaterialSubset* matls, vector<DataWarehouseP>& dws)
{
  DataWarehouse* fromDW = mapDataWarehouse(Task::OldDW, dws);
  DataWarehouse* toDW = mapDataWarehouse(Task::NewDW, dws);
  if(d_action)
     d_action->doit(pc, patches, matls, fromDW, toDW);
}

void
Task::display( ostream & out ) const
{
  out << getName() << " (" << d_tasktype << "): [";
  if( patch_set != 0 ){
    out << "Patches: {";
    for(int i=0;i<patch_set->size();i++){
      const PatchSubset* ps = patch_set->getSubset(i);
      if(i != 0)
	out << ", ";
      out << "{";
      for(int j=0;j<ps->size();j++){
	if(j != 0)
	  out << ",";
	const Patch* patch = ps->get(j);
	out << patch->getID();
      }
      out << "}";
    }
    out << "}";
  } else {
    out << "(No Patches)";
  }
  out << ", ";
  if( matl_set != 0 ){
    out << "Matls: {";
    for(int i=0;i< matl_set->size();i++){
      const MaterialSubset* ms = matl_set->getSubset(i);
      if(i != 0)
	out << ", ";
      out << "{";
      for(int j=0;j<ms->size();j++){
	if(j != 0)
	  out << ",";
	out << ms->get(j);
      }
      out << "}";
    }
    out << "}";
  } else {
    out << "(No Matls)";
  }
  out << ", DWs: ";
  for(int i=0;i<TotalDWs;i++){
    if(i != 0)
      out << ", ";
    out << dwmap[i];
  }
  out << "]";
}

std::ostream &
operator << ( std::ostream & out, const Uintah::Task::Dependency & dep )
{
  out << "[" << *(dep.var);
  if(dep.var->typeDescription()->isReductionVariable()){
    if(dep.reductionLevel) {
      out << " Level: " << dep.reductionLevel->getIndex();
    } else {
      out << " Global level";
    }
  } else {
    if( dep.patches ){
      out << " Patches: ";
      for(int i=0;i<dep.patches->size();i++){
	if(i > 0)
	  out << ",";
	out << dep.patches->get(i)->getID();
      }
    } else if(dep.reductionLevel) {
      out << " Level: " << dep.reductionLevel->getIndex();
    } else {
      out << " No patches";
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

void
Task::displayAll(ostream& out) const
{
   display(out);
   out << '\n';
   for(Task::Dependency* req = req_head; req != 0; req = req->next)
      out << "  requires: " << *req << '\n';
   for(Task::Dependency* comp = comp_head; comp != 0; comp = comp->next)
      out << "  computes: " << *comp << '\n';
   for(Task::Dependency* mod = mod_head; mod != 0; mod = mod->next)
      out << "  modifies: " << *mod << '\n';
}

void Task::setMapping(int dwmap[TotalDWs])
{
  for(int i=0;i<TotalDWs;i++)
    this->dwmap[i]=dwmap[i];
}

int Task::mapDataWarehouse(WhichDW dw) const
{
  ASSERTRANGE(dw, 0, Task::TotalDWs);
  return dwmap[dw];
}

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

ostream &
operator << (ostream &out, const Task & task)
{
  task.display( out );
  return out;
}

ostream &
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
  }
  return out;
}
