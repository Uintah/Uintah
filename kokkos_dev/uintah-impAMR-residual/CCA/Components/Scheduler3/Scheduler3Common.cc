#include <Packages/Uintah/CCA/Components/Scheduler3/Scheduler3Common.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/TaskGraph3.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/PatchBasedDataWarehouse3.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/PatchBasedDataWarehouse3P.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/DetailedTasks3.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>


#include <fstream>
#include <iostream>
#include <iomanip>
#include <errno.h>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <stdlib.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

static DebugStream dbg("Scheduler3Common", false);

Scheduler3Common::Scheduler3Common(const ProcessorGroup* myworld, Output* oport)
  : UintahParallelComponent(myworld), graph(this, myworld), m_outPort(oport),
    m_graphDoc(NULL), m_nodes(NULL)
{
  d_generation = 0;

  dts_ = 0;
  emit_taskgraph = false;
  memlogfile = 0;
  restartable = false;
  for(int i=0;i<Task::TotalDWs;i++)
    dwmap[i]=Task::InvalidDW;
  // Default mapping...
  dwmap[Task::OldDW]=0;
  dwmap[Task::NewDW]=1;
}

Scheduler3Common::~Scheduler3Common()
{
  if( dts_ )
    delete dts_;
  if(memlogfile)
    delete memlogfile;

  // list of vars used for AMR regridding
  for (unsigned i = 0; i < label_matls_.size(); i++)
    for ( label_matl_map::iterator iter = label_matls_[i].begin(); iter != label_matls_[i].end(); iter++)
      if (iter->second->removeReference())
        delete iter->second;
  
  label_matls_.clear();
}

void
Scheduler3Common::makeTaskGraphDoc(const DetailedTasks3*/* dt*/, int rank)
{
  if (!emit_taskgraph)
    return;

  if (!m_outPort->wasOutputTimestep())
    return;
  
  // make sure to release this DOMDocument after finishing emitting the nodes
  m_graphDoc = ProblemSpec::createDocument("Uintah_TaskGraph");
  
  ProblemSpecP meta = m_graphDoc->appendChild("Meta");
  meta->appendElement("username", getenv("LOGNAME"));
  time_t t = time(NULL);
  meta->appendElement("date", ctime(&t));
  
  m_nodes = m_graphDoc->appendChild("Nodes");
  //m_graphDoc->appendChild(m_nodes);
  
  ProblemSpecP edgesElement = m_graphDoc->appendChild("Edges");
  
  if (dts_) {
    dts_->emitEdges(edgesElement, rank);
  }
}

bool
Scheduler3Common::useInternalDeps()
{
  // keep track of internal dependencies only if it will emit
  // the taskgraphs (by default).
  // return emit_taskgraph;
  return false;
}

void
Scheduler3Common::emitNode( const DetailedTask3* task, 
                                 double        start,
                                 double        duration,
                                 double        execution_duration,
                                 double        execution_flops,
                                 double        communication_flops )
{  
    if (m_nodes == 0)
        return;
    
    ProblemSpecP node = m_nodes->appendChild("node");
    //m_nodes->appendChild(node);

    node->appendElement("name", task->getName());
    node->appendElement("start", start);
    node->appendElement("duration", duration);
    if (execution_duration > 0)
      node->appendElement("execution_duration", execution_duration);
    if (execution_flops > 0)
      node->appendElement("execution_flops", (long)execution_flops);
    if (communication_flops > 0)
      node->appendElement("communication_flops", (long)communication_flops);
}

void
Scheduler3Common::finalizeNodes(int process /* = 0*/)
{
    if (m_graphDoc == 0)
        return;

    if (m_outPort->wasOutputTimestep()) {
      string timestep_dir(m_outPort->getLastTimestepOutputLocation());
      
      ostringstream fname;
      fname << "/taskgraph_" << setw(5) << setfill('0') << process << ".xml";
      string file_name(timestep_dir + fname.str());
      ofstream graphfile(file_name.c_str());
      graphfile << m_graphDoc << "\n";
    }
    
    m_graphDoc->releaseDocument();
    m_graphDoc = NULL;
    m_nodes = NULL;
}

void
Scheduler3Common::problemSetup(const ProblemSpecP&)
{
   // For schedulers that need no setup
}

LoadBalancer*
Scheduler3Common::getLoadBalancer()
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   return lb;
}

void
Scheduler3Common::addTask(Task* task, const PatchSet* patches,
			 const MaterialSet* matls)
{
  // Save the DW map
  task->setMapping(dwmap);
  dbg << "adding Task: " << task->getName() << ", # patches: ";
  if( patches ) dbg << patches->size();
  else          dbg << "0";
  dbg << ", # matls: " ;
  if( matls ) dbg << matls->size();
  else          dbg << "0";
  dbg << "\n";

  graph.addTask(task, patches, matls);

  for (Task::Dependency* dep = task->getRequires(); dep != 0;
       dep = dep->next) {
    // Store the ghost cell information of each of the requires
    // so we can predict the total allocation needed for each variable.
    if (dep->numGhostCells > 0) {
      constHandle<PatchSubset> dep_patches = dep->getPatchesUnderDomain(patches->getUnion());
      constHandle<MaterialSubset> dep_matls = dep->getMaterialsUnderDomain(matls->getUnion());
      m_ghostOffsetVarMap.includeOffsets(dep->var, dep_matls.get_rep(), dep_patches.get_rep(),
					 dep->gtype, dep->numGhostCells);
    }
  }
}

void
Scheduler3Common::releaseLoadBalancer()
{
  releasePort("load balancer");
}

void
Scheduler3Common::initialize(int numOldDW /* =1 */, int numNewDW /* =1 */,
			    DataWarehouse* parent_old_dw /* =0 */,
			    DataWarehouse* parent_new_dw /* =0 */)
{

  // doesn't really do anything except initialize/clear the taskgraph
  //   if the default parameter values are used
  int numDW = numOldDW+numNewDW;
  int oldnum = (int)dws.size();
  // Clear out the data warehouse so that memory will be freed
  for(int i=numDW;i<oldnum;i++)
    dws[i]=0;
  dws.resize(numDW);
  for(;oldnum < numDW; oldnum++)
    dws[oldnum] = 0;
  numOldDWs = numOldDW;
  graph.initialize();
  m_ghostOffsetVarMap.clear();
  PatchBasedDataWarehouse3* pold = dynamic_cast<PatchBasedDataWarehouse3*>(parent_old_dw);
  PatchBasedDataWarehouse3* pnew = dynamic_cast<PatchBasedDataWarehouse3*>(parent_new_dw);
  if(parent_old_dw && parent_new_dw){
    ASSERT(pold != 0);
    ASSERT(pnew != 0);
    ASSERT(numOldDW > 2);
    dws[0]=pold;
    dws[1]=pnew;
  }
}

void Scheduler3Common::clearMappings()
{
  for(int i=0;i<Task::TotalDWs;i++)
    dwmap[i]=-1;
}

void Scheduler3Common::mapDataWarehouse(Task::WhichDW which, int dwTag)
{
  ASSERTRANGE(which, 0, Task::TotalDWs);
  ASSERTRANGE(dwTag, 0, (int)dws.size());
  dwmap[which]=dwTag;
}

DataWarehouse*
Scheduler3Common::get_dw(int idx)
{
  ASSERTRANGE(idx, 0, (int)dws.size());
  return dws[idx].get_rep();
}

DataWarehouse*
Scheduler3Common::getLastDW(void)
{
  return get_dw(static_cast<int>(dws.size()) - 1);
}

void 
Scheduler3Common::advanceDataWarehouse(const GridP& grid)
{
  dbg << "advanceDataWarehouse, numDWs = " << dws.size() << '\n';
  ASSERT(dws.size() >= 2);
  // The last becomes last old, and the rest are new
  dws[numOldDWs-1] = dws[dws.size()-1];
  if (dws.size() == 2 && dws[0] == 0) {
    // first datawarehouse -- indicate that it is the "initialization" dw.
    int generation = d_generation++;
    dws[1] = scinew PatchBasedDataWarehouse3(d_myworld, this, generation, grid,
					  true /* initialization dw */);
  } else {
    for(int i=numOldDWs;i<(int)dws.size();i++) {
      replaceDataWarehouse(i, grid);
    }
  }
}

void Scheduler3Common::fillDataWarehouses(const GridP& grid)
{
  for(int i=numOldDWs;i<(int)dws.size();i++)
    if(!dws[i])
      replaceDataWarehouse(i, grid);
  d_generation++; // should this happen here? Bryan
}

void Scheduler3Common::replaceDataWarehouse(int index, const GridP& grid)
{
  dws[index] = scinew PatchBasedDataWarehouse3(d_myworld, this, d_generation++, grid);
}

void Scheduler3Common::setRestartable(bool restartable)
{
  this->restartable = restartable;
}

const vector<const Patch*>* Scheduler3Common::
getSuperPatchExtents(const VarLabel* label, int matlIndex, const Patch* patch,
                     Ghost::GhostType requestedGType, int requestedNumGCells,
                     IntVector& requiredLow, IntVector& requiredHigh,
                     IntVector& requestedLow, IntVector& requestedHigh) const
{
  const SuperPatch* connectedPatchGroup =
    m_locallyComputedPatchVarMap.getConnectedPatchGroup(label, patch);
  if (connectedPatchGroup == 0)
    return 0;

  SuperPatch::Region requestedExtents = connectedPatchGroup->getRegion();
  SuperPatch::Region requiredExtents = connectedPatchGroup->getRegion();  
  
  // expand to cover the entire connected patch group
  bool containsGivenPatch = false;
  for (unsigned int i = 0; i < connectedPatchGroup->getBoxes().size(); i++) {
    // get the minimum extents containing both the expected ghost cells
    // to be needed and the given ghost cells.
    const Patch* memberPatch = connectedPatchGroup->getBoxes()[i];
    VarLabelMatlPatch vmp(label, matlIndex, memberPatch);
    m_ghostOffsetVarMap.getExtents(vmp, requestedGType, requestedNumGCells,
                                   requiredLow, requiredHigh,
                                   requestedLow, requestedHigh);
    SuperPatch::Region requiredRegion =
      SuperPatch::Region(requiredLow, requiredHigh);
    requiredExtents = requiredExtents.enclosingRegion(requiredRegion);
    SuperPatch::Region requestedRegion =
      SuperPatch::Region(requestedLow, requestedHigh);
    requestedExtents = requestedExtents.enclosingRegion(requestedRegion);
    if (memberPatch == patch)
      containsGivenPatch = true;
  }
  ASSERT(containsGivenPatch);
  
  requiredLow = requiredExtents.low_;
  requiredHigh = requiredExtents.high_;
  requestedLow = requestedExtents.low_;
  requestedHigh = requestedExtents.high_;

  // requested extents must enclose the required extents at lesst.
  ASSERTEQ(Min(requiredLow, requestedLow), requestedLow);
  ASSERTEQ(Max(requiredHigh, requestedHigh), requestedHigh);
  
  return &connectedPatchGroup->getBoxes();
}

void
Scheduler3Common::logMemoryUse()
{
  if(!memlogfile){
    ostringstream fname;
    fname << "uintah_memuse.log.p" << setw(5) << setfill('0') << d_myworld->myrank();
    memlogfile = new ofstream(fname.str().c_str());
    if(!*memlogfile){
      cerr << "Error opening file: " << fname.str() << '\n';
    }
  }
  *memlogfile << '\n';
  unsigned long total = 0;
  for(int i=0;i<(int)dws.size();i++){
    char* name;
    if(i==0)
      name="OldDW";
    else if(i==(int)dws.size()-1)
      name="NewDW";
    else
      name="IntermediateDW";
    dws[i]->logMemoryUse(*memlogfile, total, name);
  }
  if(dts_)
    dts_->logMemoryUse(*memlogfile, total, "Taskgraph");
  *memlogfile << "Total: " << total << '\n';
  memlogfile->flush();
}

// Makes and returns a map that maps strings to VarLabels of
// that name and a list of material indices for which that
// variable is valid (according to d_allcomps in graph).
Scheduler::VarLabelMaterialMap* Scheduler3Common::makeVarLabelMaterialMap()
{
  return graph.makeVarLabelMaterialMap();
}
     
const vector<const Task::Dependency*>& Scheduler3Common::getInitialRequires()
{
  return graph.getInitialRequires();
}

void Scheduler3Common::doEmitTaskGraphDocs()
{
  emit_taskgraph=true;
}

void Scheduler3Common::compile()
{
  actuallyCompile();
  m_locallyComputedPatchVarMap.reset();

  if (dts_ != 0) {

    dts_->computeLocalTasks(d_myworld->myrank());
    dts_->createScrubCounts();
    
    // figure out the locally computed patches for each variable.
    for (int i = 0; i < dts_->numLocalTasks(); i++) {
      const DetailedTask3* dt = dts_->localTask(i);
      for(const Task::Dependency* comp = dt->getTask()->getComputes();
	  comp != 0; comp = comp->next){
	constHandle<PatchSubset> patches =
	  comp->getPatchesUnderDomain(dt->getPatches());
	m_locallyComputedPatchVarMap.addComputedPatchSet(comp->var,
                                                         patches.get_rep());
      }
    }
  }
  m_locallyComputedPatchVarMap.makeGroups();
}

bool Scheduler3Common::isOldDW(int idx) const
{
  ASSERTRANGE(idx, 0, static_cast<int>(dws.size()));
  return idx < numOldDWs;
}

bool Scheduler3Common::isNewDW(int idx) const
{
  ASSERTRANGE(idx, 0, static_cast<int>(dws.size()));
  return idx >= numOldDWs;
}

void
Scheduler3Common::finalizeTimestep()
{
  finalizeNodes(d_myworld->myrank());
  for(unsigned int i=numOldDWs;i<dws.size();i++)
    dws[i]->finalize();
}

void
Scheduler3Common::scheduleAndDoDataCopy(const GridP& grid, SimulationStateP& state, 
                                       SimulationInterface* sim)
{
  // clear the old list of vars and matls
  for (unsigned i = 0; i < label_matls_.size(); i++)
    for ( label_matl_map::iterator iter = label_matls_[i].begin(); iter != label_matls_[i].end(); iter++)
      if (iter->second->removeReference())
        delete iter->second;
  
  label_matls_.clear();
  label_matls_.resize(grid->numLevels());

  // produce a map from all tasks' requires from the Old DW.  Store the varlabel and matls
  for (int i = 0; i < graph.getNumTasks(); i++) {
    for(Task::Dependency* dep = graph.getTask(i)->getRequires(); dep != 0; dep=dep->next){
      if(this->isOldDW(dep->mapDataWarehouse())) {
        if (dep->var->typeDescription()->getType() == TypeDescription::ReductionVariable)
          // we will take care of reduction variables in a different section
          continue;

        // check the level on the case where variables are only computed on certain levels
        const PatchSet* ps = graph.getTask(i)->getPatchSet();
        int level = -1;
        if (ps) 
          if (ps->getSubset(0))
            level = getLevel(ps->getSubset(0))->getIndex();
        
        if (level == -1)
          // shouldn't really happen...
          continue;

        const MaterialSubset* matSubset = (dep->matls != 0) ?
          dep->matls : dep->task->getMaterialSet()->getUnion();


        // if var was already found, make a union of the materials
        MaterialSubset* matls = scinew MaterialSubset(matSubset->getVector());
        matls->addReference();
        
        MaterialSubset* union_matls;
        union_matls = label_matls_[level][dep->var];
        if (union_matls) {
          for (int i = 0; i < union_matls->size(); i++) 
            if (!matls->contains(union_matls->get(i)))
              matls->add(union_matls->get(i));
        }
        matls->sort();        
        label_matls_[level][dep->var] = matls;
      }
    }
  }
  
  this->advanceDataWarehouse(grid);
  this->initialize(1, 1);
  this->clearMappings();
  this->mapDataWarehouse(Task::OldDW, 0);
  this->mapDataWarehouse(Task::NewDW, 1);
  
  DataWarehouse* oldDataWarehouse = this->get_dw(0);
  DataWarehouse* newDataWarehouse = this->getLastDW();
  oldDataWarehouse->setScrubbing(DataWarehouse::ScrubNone);
  newDataWarehouse->setScrubbing(DataWarehouse::ScrubNone);
  const Grid* oldGrid = oldDataWarehouse->getGrid();
  vector<Task*> dataTasks;
  SchedulerP sched(dynamic_cast<Scheduler*>(this));

  for (int i = 0; i < grid->numLevels(); i++) {
    LevelP newLevel = newDataWarehouse->getGrid()->getLevel(i);

    if (i > 0) {

      PatchSet* refineSet = scinew PatchSet;
      if (i >= oldGrid->numLevels())
        refineSet = const_cast<PatchSet*>(newLevel->eachPatch());
      else {
        refineSet = scinew PatchSet;
        LevelP oldLevel = oldDataWarehouse->getGrid()->getLevel(newLevel->getIndex());
        
        // go through the patches, and find if there are patches that weren't entirely 
        // covered by patches on the old grid, and interpolate them.  
        // then after, copy the data, and if necessary, overwrite interpolated data
        
        for (Level::patchIterator iter = newLevel->patchesBegin(); iter != newLevel->patchesEnd(); iter++) {
          Patch* newPatch = *iter;
          
          // get the low/high for what we'll need to get
          IntVector lowIndex, highIndex;
          newPatch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0), Ghost::None, 0, lowIndex, highIndex);
          
          // find if area on the new patch was not covered by the old patches
          IntVector dist = highIndex-lowIndex;
          int totalCells = dist.x()*dist.y()*dist.z();
          int sum = 0;
          Patch::selectType oldPatches;
          oldLevel->selectPatches(lowIndex, highIndex, oldPatches);
          
          for (int old = 0; old < oldPatches.size(); old++) {
            const Patch* oldPatch = oldPatches[old];
            IntVector low = Max(oldPatch->getLowIndex(), newPatch->getLowIndex());
            IntVector high = Min(oldPatch->getHighIndex(), newPatch->getHighIndex());
            IntVector dist = high-low;
            sum += dist.x()*dist.y()*dist.z();
          }  // for oldPatches
          if (sum != totalCells) {
            refineSet->add(newPatch);
          }
          
        } // for patchIterator
      }
      if (refineSet->size() > 0)
        sim->scheduleRefine(refineSet, sched);
    }
    
    dataTasks.push_back(scinew Task("Scheduler3Common::copyDataToNewGrid", this,                          
                                     &Scheduler3Common::copyDataToNewGrid));
    addTask(dataTasks[i], newLevel->eachPatch(), state->allMaterials());

    for ( label_matl_map::iterator iter = label_matls_[i].begin(); iter != label_matls_[i].end(); iter++) {
      const VarLabel* var = iter->first;
      MaterialSubset* matls = iter->second;
      
      dataTasks[i]->requires(Task::OldDW, var, 0, Task::OtherGridDomain, matls, Task::NormalDomain, Ghost::None, 0);
      dataTasks[i]->computes(var, matls);
    }
  }

  // set so the load balancer will make an adequate neighborhood, as the default
  // neighborhood isn't good enough for the copy data timestep
  state->setCopyDataTimestep(true);

  this->compile(); 
  this->execute();
  
  state->setCopyDataTimestep(false);

  vector<VarLabelMatl<Level> > levelVariableInfo;
  oldDataWarehouse->getVarLabelMatlLevelTriples(levelVariableInfo);
  
  // copy reduction variables
  
  newDataWarehouse->unfinalize();
  for ( unsigned int i = 0; i < levelVariableInfo.size(); i++ ) {
    VarLabelMatl<Level> currentReductionVar = levelVariableInfo[i];
    // cout << "REDUNCTION:  Label(" << setw(15) << currentReductionVar.label_->getName() << "): Patch(" << reinterpret_cast<int>(currentReductionVar.level_) << "): Material(" << currentReductionVar.matlIndex_ << ")" << endl; 
    const Level* oldLevel = currentReductionVar.domain_;
    const Level* newLevel = NULL;
    if (oldLevel) {
      newLevel = (newDataWarehouse->getGrid()->getLevel( oldLevel->getIndex() )).get_rep();
    }
    
    ReductionVariableBase* v = dynamic_cast<ReductionVariableBase*>(currentReductionVar.label_->typeDescription()->createInstance());
    oldDataWarehouse->get(*v, currentReductionVar.label_, currentReductionVar.domain_, currentReductionVar.matlIndex_);
    newDataWarehouse->put(*v, currentReductionVar.label_, newLevel, currentReductionVar.matlIndex_);
  }
  newDataWarehouse->refinalize();
}


void
Scheduler3Common::copyDataToNewGrid(const ProcessorGroup*, const PatchSubset* patches,
                                   const MaterialSubset* matls, DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  dbg << "Scheduler3Common::copyDataToNewGrid() BGN" << endl;

  PatchBasedDataWarehouse3* oldDataWarehouse = dynamic_cast<PatchBasedDataWarehouse3*>(old_dw);
  PatchBasedDataWarehouse3* newDataWarehouse = dynamic_cast<PatchBasedDataWarehouse3*>(new_dw);

  // For each patch in the patch subset which contains patches in the new grid

  for ( int idx = 0; idx < patches->size(); idx++ ) {
    //cerr << "Patches[ " << idx << " ] = " << patches->get(idx)->getID() << endl;
  }

  for ( int p = 0; p < patches->size(); p++ ) {
    const Patch* newPatch = patches->get(p);
    const Level* newLevel = newPatch->getLevel();
    
    // If there is a level that didn't exist, we don't need to copy it
    if ( newLevel->getIndex() >= oldDataWarehouse->getGrid()->numLevels() ) {
      continue;
    }
    
    // find old patches associated with this patch
    const Level* oldLevel = (oldDataWarehouse->getGrid()->getLevel( newLevel->getIndex() )).get_rep();
    
    for ( label_matl_map::iterator iter = label_matls_[oldLevel->getIndex()].begin(); 
          iter != label_matls_[oldLevel->getIndex()].end(); iter++) {
      const VarLabel* label = iter->first;
      MaterialSubset* var_matls = iter->second;
         
      // get the low/high for what we'll need to get
      Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), true);
      IntVector newLowIndex, newHighIndex;
      newPatch->computeVariableExtents(basis, label->getBoundaryLayer(), Ghost::AroundCells, 0, newLowIndex, newHighIndex);

      for (int m = 0; m < var_matls->size(); m++) {

        int matl = var_matls->get(m);
        if (!matls->contains(matl)) {
          //cout << "We are skipping material " << currentVar.matlIndex_ << endl;
          continue;
        }

        Patch::selectType oldPatches;
        oldLevel->selectPatches(newLowIndex, newHighIndex, oldPatches);
        
        for ( int oldIdx = 0;  oldIdx < oldPatches.size(); oldIdx++) {
          const Patch* oldPatch = oldPatches[oldIdx];
          
          IntVector oldLowIndex = oldPatch->getLowIndex(basis, label->getBoundaryLayer());
          IntVector oldHighIndex = oldPatch->getHighIndex(basis, label->getBoundaryLayer());
          
          IntVector copyLowIndex = Max(newLowIndex, oldLowIndex);
          IntVector copyHighIndex = Min(newHighIndex, oldHighIndex);
        
          // based on the selectPatches above, we might have patches we don't want to use, so prune them here.
          if (copyLowIndex.x() >= copyHighIndex.x() || copyLowIndex.y() >= copyHighIndex.y() || copyLowIndex.z() >= copyHighIndex.z())
            continue;
          
          switch(label->typeDescription()->getType()){
          case TypeDescription::NCVariable:
          case TypeDescription::CCVariable:
          case TypeDescription::SFCXVariable:
          case TypeDescription::SFCYVariable:
          case TypeDescription::SFCZVariable:
            {
              if(!oldDataWarehouse->exists(label, matl, oldPatch))
                SCI_THROW(UnknownVariable(label->getName(), oldDataWarehouse->getID(), oldPatch, matl,
                                          "in copyDataTo NCVariable"));
              GridVariable* v = dynamic_cast<GridVariable*>(oldDataWarehouse->d_varDB.get(label, matl, oldPatch));
              
              if ( !newDataWarehouse->d_varDB.exists(label, matl, newPatch) ) {
                GridVariable* newVariable = v->cloneType();
                newVariable->rewindow( newLowIndex, newHighIndex );
                newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
                newDataWarehouse->d_varDB.put(label, matl, newPatch, newVariable, false);
              } else {
                GridVariable* newVariable = dynamic_cast<GridVariable*>(newDataWarehouse->d_varDB.get(label, matl, newPatch ));
                // make sure it exists in the right region (it might be ghost data)
                newVariable->rewindow(newLowIndex, newHighIndex);
                if (oldPatch->isVirtual()) {
                  // it can happen where the old patch was virtual and this is not
                  NCVariableBase* tmpVar = dynamic_cast<NCVariableBase*>(newVariable->cloneType());
                  oldDataWarehouse->d_varDB.get(label, matl, oldPatch, *tmpVar);
                  tmpVar->offset(oldPatch->getVirtualOffset());
                  newVariable->copyPatch( tmpVar, copyLowIndex, copyHighIndex );
                  delete tmpVar;
                }
                else
                  newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
              }
            }
            break;
          case TypeDescription::ParticleVariable:
            {
              if(!oldDataWarehouse->d_varDB.exists(label, matl, oldPatch))
                SCI_THROW(UnknownVariable(label->getName(), oldDataWarehouse->getID(), oldPatch, matl,
                                          "in copyDataTo ParticleVariable"));
              if ( !newDataWarehouse->d_varDB.exists(label, matl, newPatch) ) {
                PatchSubset* ps = new PatchSubset;
                ps->add(oldPatch);
                PatchSubset* newps = new PatchSubset;
                newps->add(newPatch);
                MaterialSubset* ms = new MaterialSubset;
                ms->add(matl);
                newDataWarehouse->transferFrom(oldDataWarehouse, label, ps, ms, false, newps);
                delete ps;
                delete ms;
                delete newps;
              } else {
                cout << "Particle copy not implemented for pre-existent var (BNR Regridder?)\n";
                SCI_THROW(UnknownVariable(label->getName(), newDataWarehouse->getID(), oldPatch, matl,
                                          "in copyDataTo ParticleVariable"));
              }
            }
            break;
          case TypeDescription::PerPatch:
            {
            }
            break;
          default:
            SCI_THROW(InternalError("Unknown variable type in transferFrom: "+label->getName()));
          } // end switch
        } // end oldPatches
      } // end matls
    } // end label_matls
  } // end patches

  // d_lock.writeUnlock(); Do we need this?

  dbg << "Scheduler3Common::copyDataToNewGrid() END" << endl;
}
