#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
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
#include <stdlib.h>

using namespace Uintah;
using namespace SCIRun;

using std::cerr;
using std::string;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

static DebugStream dbg("SchedulerCommon", false);

SchedulerCommon::SchedulerCommon(const ProcessorGroup* myworld, Output* oport)
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

SchedulerCommon::~SchedulerCommon()
{
  if( dts_ )
    delete dts_;
  if(memlogfile)
    delete memlogfile;
}

void
SchedulerCommon::makeTaskGraphDoc(const DetailedTasks*/* dt*/, int rank)
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
SchedulerCommon::useInternalDeps()
{
  // keep track of internal dependencies only if it will emit
  // the taskgraphs (by default).
  // return emit_taskgraph;
  return false;
}

void
SchedulerCommon::emitNode( const DetailedTask* task, 
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
SchedulerCommon::finalizeNodes(int process /* = 0*/)
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
SchedulerCommon::problemSetup(const ProblemSpecP&)
{
   // For schedulers that need no setup
}

LoadBalancer*
SchedulerCommon::getLoadBalancer()
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   return lb;
}

void
SchedulerCommon::addTask(Task* task, const PatchSet* patches,
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
SchedulerCommon::releaseLoadBalancer()
{
  releasePort("load balancer");
}

void
SchedulerCommon::initialize(int numOldDW /* =1 */, int numNewDW /* =1 */,
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
  OnDemandDataWarehouse* pold = dynamic_cast<OnDemandDataWarehouse*>(parent_old_dw);
  OnDemandDataWarehouse* pnew = dynamic_cast<OnDemandDataWarehouse*>(parent_new_dw);
  if(parent_old_dw && parent_new_dw){
    ASSERT(pold != 0);
    ASSERT(pnew != 0);
    ASSERT(numOldDW > 2);
    dws[0]=pold;
    dws[1]=pnew;
  }
}

void SchedulerCommon::clearMappings()
{
  for(int i=0;i<Task::TotalDWs;i++)
    dwmap[i]=-1;
}

void SchedulerCommon::mapDataWarehouse(Task::WhichDW which, int dwTag)
{
  ASSERTRANGE(which, 0, Task::TotalDWs);
  ASSERTRANGE(dwTag, 0, (int)dws.size());
  dwmap[which]=dwTag;
}

DataWarehouse*
SchedulerCommon::get_dw(int idx)
{
  ASSERTRANGE(idx, 0, (int)dws.size());
  return dws[idx].get_rep();
}

DataWarehouse*
SchedulerCommon::getLastDW(void)
{
  return get_dw(static_cast<int>(dws.size()) - 1);
}

void 
SchedulerCommon::advanceDataWarehouse(const GridP& grid)
{
  dbg << "advanceDataWarehouse, numDWs = " << dws.size() << '\n';
  ASSERT(dws.size() >= 2);
  // The last becomes last old, and the rest are new
  dws[numOldDWs-1] = dws[dws.size()-1];
  if (dws.size() == 2 && dws[0] == 0) {
    // first datawarehouse -- indicate that it is the "initialization" dw.
    int generation = d_generation++;
    dws[1] = scinew OnDemandDataWarehouse(d_myworld, this, generation, grid,
					  true /* initialization dw */);
  } else {
    for(int i=numOldDWs;i<(int)dws.size();i++) {
      replaceDataWarehouse(i, grid);
    }
  }
}

void SchedulerCommon::fillDataWarehouses(const GridP& grid)
{
  for(int i=numOldDWs;i<(int)dws.size();i++)
    if(!dws[i])
      replaceDataWarehouse(i, grid);
  d_generation++; // should this happen here? Bryan
}

void SchedulerCommon::replaceDataWarehouse(int index, const GridP& grid)
{
  dws[index] = scinew OnDemandDataWarehouse(d_myworld, this, d_generation++, grid);
}

void SchedulerCommon::setRestartable(bool restartable)
{
  this->restartable = restartable;
}

const vector<const Patch*>* SchedulerCommon::
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
SchedulerCommon::logMemoryUse()
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
Scheduler::VarLabelMaterialMap* SchedulerCommon::makeVarLabelMaterialMap()
{
  return graph.makeVarLabelMaterialMap();
}
     
const vector<const Task::Dependency*>& SchedulerCommon::getInitialRequires()
{
  return graph.getInitialRequires();
}

void SchedulerCommon::doEmitTaskGraphDocs()
{
  emit_taskgraph=true;
}

void SchedulerCommon::compile()
{
  actuallyCompile();
  m_locallyComputedPatchVarMap.reset();

  if (dts_ != 0) {

    dts_->computeLocalTasks(d_myworld->myrank());
    dts_->createScrubCounts();
    
    // figure out the locally computed patches for each variable.
    for (int i = 0; i < dts_->numLocalTasks(); i++) {
      const DetailedTask* dt = dts_->localTask(i);
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

bool SchedulerCommon::isOldDW(int idx) const
{
  ASSERTRANGE(idx, 0, static_cast<int>(dws.size()));
  return idx < numOldDWs;
}

bool SchedulerCommon::isNewDW(int idx) const
{
  ASSERTRANGE(idx, 0, static_cast<int>(dws.size()));
  return idx >= numOldDWs;
}

void
SchedulerCommon::finalizeTimestep()
{
  finalizeNodes(d_myworld->myrank());
  for(unsigned int i=numOldDWs;i<dws.size();i++)
    dws[i]->finalize();
}

void
SchedulerCommon::copyDataToNewGrid(const ProcessorGroup*, const PatchSubset* patches,
		  const MaterialSubset* matls, DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  dbg << "SchedulerCommon::copyDataToNewGrid() BGN" << endl;

  OnDemandDataWarehouse* oldDataWarehouse = dynamic_cast<OnDemandDataWarehouse*>(old_dw);
  OnDemandDataWarehouse* newDataWarehouse = dynamic_cast<OnDemandDataWarehouse*>(new_dw);

  // For each patch in the patch subset which contains patches in the new grid

  for ( int idx = 0; idx < patches->size(); idx++ ) {
    //cerr << "Patches[ " << idx << " ] = " << patches->get(idx)->getID() << endl;
  }

  for ( int idx = 0; idx < matls->size(); idx++ ) {
    //cerr << "Matls[ " << idx << " ] = " << matls->get(idx) << endl;
  }

  vector<VarLabelMatlPatch> variableInfo;
  oldDataWarehouse->getVarLabelMatlPatchTriples(variableInfo);

  for ( unsigned int i = 0; i < variableInfo.size(); i++ ) {

    VarLabelMatlPatch currentVar = variableInfo[i];

    if (!matls->contains(currentVar.matlIndex_)) {
      //cout << "We are skipping material " << currentVar.matlIndex_ << endl;
      continue;
    }

    //    cerr << "RANDY: SchedulerCommon::copyDataToNewGrid() Copying (" << *currentVar.label_;
    //    cerr << ") on patch(" << currentVar.patch_->getID() << ") on matl(" << currentVar.matlIndex_ << ")" << endl;
    //cout << "  Label(" << setw(15) << currentVar.label_->getName() << "): Patch(" << currentVar.patch_->getID() << "): Material(" << currentVar.matlIndex_ << ")" << endl; 
    const Level* oldLevel = currentVar.patch_->getLevel();

    // If there is a level that no longer exists, we don't need to copy it
    if ( oldLevel->getIndex() >= newDataWarehouse->getGrid()->numLevels() ) {
      continue;
    }

    const Level* newLevel = (newDataWarehouse->getGrid()->getLevel( oldLevel->getIndex() )).get_rep();
    const Patch* oldPatch = currentVar.patch_;

    IntVector lowIndex, highIndex;
    Patch::VariableBasis basis = Patch::translateTypeToBasis(currentVar.label_->typeDescription()->getType(), true);
    oldPatch->computeVariableExtents(basis, currentVar.label_->getBoundaryLayer(), Ghost::AroundCells, 0, lowIndex, highIndex);
    //cout << "  oldPatch(" << oldPatch->getID() << ") = " << lowIndex << " to " << highIndex << endl;
    Patch::selectType neighbors;
    newLevel->selectPatches(lowIndex, highIndex, neighbors);

    for ( int newPatchIndex = 0; newPatchIndex < neighbors.size(); newPatchIndex++) {

      const Patch* newPatch = neighbors[newPatchIndex];

      if (!patches->contains(newPatch)) {
	//cout << "We are skipping patch " << newPatch->getID() << endl;
	continue;
      }

      //      newDataWarehouse->checkPutAccess(currentVar.label_, currentVar.matlIndex_, currentVar.patch_, false);      
      switch(currentVar.label_->typeDescription()->getType()){
      case TypeDescription::NCVariable:
	{
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH NCVARIABLE BGN" << endl;
	  if(!oldDataWarehouse->d_ncDB.exists(currentVar.label_, currentVar.matlIndex_, currentVar.patch_))
	    SCI_THROW(UnknownVariable(currentVar.label_->getName(), oldDataWarehouse->getID(), currentVar.patch_, currentVar.matlIndex_,
				      "in copyDataTo NCVariable"));
	  NCVariableBase* v = oldDataWarehouse->d_ncDB.get(currentVar.label_, currentVar.matlIndex_, currentVar.patch_);

	  IntVector newLowIndex = newPatch->getLowIndex(basis, currentVar.label_->getBoundaryLayer());
	  IntVector newHighIndex = newPatch->getHighIndex(basis, currentVar.label_->getBoundaryLayer());
	  //cout << "  newPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  IntVector copyLowIndex = Max(newLowIndex, lowIndex);
	  IntVector copyHighIndex = Min(newHighIndex, highIndex);
	  //cout << "  copPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  //cout << "Var Low  = " << v->getLow() << endl;
	  //cout << "Var High = " << v->getHigh() << endl;
	  /*
	  for(CellIterator iter(v->getLow(), v->getHigh()); !iter.done(); iter++){
	    //cout << "RANDY: v" << *iter << " = " << (*v)[*iter] << endl;
	  }
	  */
	  if ( !newDataWarehouse->d_ncDB.exists(currentVar.label_, currentVar.matlIndex_, newPatch) ) {
	    NCVariableBase* newVariable = v->cloneType();
	    newVariable->rewindow( newLowIndex, newHighIndex );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	    //	    for(CellIterator iter(v->getLow(), v->getHigh()); !iter.done(); iter++){
	    //	      cout << "RANDY: newv" << *iter << " = " << v[*iter] << endl;
	    //	    }

	    newDataWarehouse->d_ncDB.put(currentVar.label_, currentVar.matlIndex_, newPatch, newVariable, false);
	  } else {
	    NCVariableBase* newVariable = newDataWarehouse->d_ncDB.get(currentVar.label_, currentVar.matlIndex_, newPatch );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	    //	    for(CellIterator iter(v->getLow(), v->getHigh()); !iter.done(); iter++){
	    //	      cout << "RANDY: newv2" << *iter << " = " << v[*iter] << endl;
	    //	    }
	  }
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH NCVARIABLE END" << endl;
	}
	break;
      case TypeDescription::CCVariable:
	{
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH CCVARIABLE BGN" << endl;
	  if(!oldDataWarehouse->d_ccDB.exists(currentVar.label_, currentVar.matlIndex_, currentVar.patch_))
	    SCI_THROW(UnknownVariable(currentVar.label_->getName(), oldDataWarehouse->getID(), currentVar.patch_, currentVar.matlIndex_,
				      "in copyDataTo CCVariable"));

	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH CCVARIABLE AAA" << endl;

	  CCVariableBase* v = oldDataWarehouse->d_ccDB.get(currentVar.label_, currentVar.matlIndex_, currentVar.patch_);

	  IntVector newLowIndex = newPatch->getLowIndex(basis, currentVar.label_->getBoundaryLayer());
	  IntVector newHighIndex = newPatch->getHighIndex(basis, currentVar.label_->getBoundaryLayer());
	  //cout << "  newPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  IntVector copyLowIndex = Max(newLowIndex, lowIndex);
	  IntVector copyHighIndex = Min(newHighIndex, highIndex);
	  //cout << "  copPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  //cout << "Var Low  = " << v->getLow() << endl;
	  //cout << "Var High = " << v->getHigh() << endl;
	  if ( !newDataWarehouse->d_ccDB.exists(currentVar.label_, currentVar.matlIndex_, newPatch) ) {
	    //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH CCVARIABLE BBB" << endl;
	    CCVariableBase* newVariable = v->cloneType();
	    newVariable->rewindow( newLowIndex, newHighIndex );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	    newDataWarehouse->d_ccDB.put(currentVar.label_, currentVar.matlIndex_, newPatch, newVariable, false);
	  } else {
	    //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH CCVARIABLE CCC" << endl;
	    CCVariableBase* newVariable = newDataWarehouse->d_ccDB.get(currentVar.label_, currentVar.matlIndex_, newPatch );
 	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	  }
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH CCVARIABLE END" << endl;
	}
	break;
      case TypeDescription::SFCXVariable:
	{
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH SFCXVARIABLE BGN" << endl;
	  if(!oldDataWarehouse->d_sfcxDB.exists(currentVar.label_, currentVar.matlIndex_, currentVar.patch_))
	    SCI_THROW(UnknownVariable(currentVar.label_->getName(), oldDataWarehouse->getID(), currentVar.patch_, currentVar.matlIndex_,
				      "in copyDataTo SFCXVariable"));

	  SFCXVariableBase* v = oldDataWarehouse->d_sfcxDB.get(currentVar.label_, currentVar.matlIndex_, currentVar.patch_);

	  IntVector newLowIndex = newPatch->getLowIndex(basis, currentVar.label_->getBoundaryLayer());
	  IntVector newHighIndex = newPatch->getHighIndex(basis, currentVar.label_->getBoundaryLayer());
	  //cout << "  newPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  IntVector copyLowIndex = Max(newLowIndex, lowIndex);
	  IntVector copyHighIndex = Min(newHighIndex, highIndex);
	  //cout << "  copPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  //cout << "Var Low  = " << v->getLow() << endl;
	  //cout << "Var High = " << v->getHigh() << endl;
	  if ( !newDataWarehouse->d_sfcxDB.exists(currentVar.label_, currentVar.matlIndex_, newPatch) ) {
	    SFCXVariableBase* newVariable = v->cloneType();
	    newVariable->rewindow( newLowIndex, newHighIndex );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	    newDataWarehouse->d_sfcxDB.put(currentVar.label_, currentVar.matlIndex_, newPatch, newVariable, false);
	  } else {
	    SFCXVariableBase* newVariable = newDataWarehouse->d_sfcxDB.get(currentVar.label_, currentVar.matlIndex_, newPatch );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	  }
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH SFCXVARIABLE END" << endl;
	}
	break;
      case TypeDescription::SFCYVariable:
	{
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH SFCYVARIABLE BGN " << endl;
	  if(!oldDataWarehouse->d_sfcyDB.exists(currentVar.label_, currentVar.matlIndex_, currentVar.patch_))
	    SCI_THROW(UnknownVariable(currentVar.label_->getName(), oldDataWarehouse->getID(), currentVar.patch_, currentVar.matlIndex_,
				      "in copyDataTo SFCYVariable"));

	  SFCYVariableBase* v = oldDataWarehouse->d_sfcyDB.get(currentVar.label_, currentVar.matlIndex_, currentVar.patch_);

	  IntVector newLowIndex = newPatch->getLowIndex(basis, currentVar.label_->getBoundaryLayer());
	  IntVector newHighIndex = newPatch->getHighIndex(basis, currentVar.label_->getBoundaryLayer());
	  //cout << "  newPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  IntVector copyLowIndex = Max(newLowIndex, lowIndex);
	  IntVector copyHighIndex = Min(newHighIndex, highIndex);
	  //cout << "  copPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  //cout << "Var Low  = " << v->getLow() << endl;
	  //cout << "Var High = " << v->getHigh() << endl;
	  if ( !newDataWarehouse->d_sfcyDB.exists(currentVar.label_, currentVar.matlIndex_, newPatch) ) {
	    SFCYVariableBase* newVariable = v->cloneType();
	    newVariable->rewindow( newLowIndex, newHighIndex );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	    newDataWarehouse->d_sfcyDB.put(currentVar.label_, currentVar.matlIndex_, newPatch, newVariable, false);
	  } else {
	    SFCYVariableBase* newVariable = newDataWarehouse->d_sfcyDB.get(currentVar.label_, currentVar.matlIndex_, newPatch );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	  }
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH SFCYVARIABLE END " << endl;
	}
	break;
      case TypeDescription::SFCZVariable:
	{
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH SFCZVARIABLE BGN" << endl;
	  if(!oldDataWarehouse->d_sfczDB.exists(currentVar.label_, currentVar.matlIndex_, currentVar.patch_))
	    SCI_THROW(UnknownVariable(currentVar.label_->getName(), oldDataWarehouse->getID(), currentVar.patch_, currentVar.matlIndex_,
				      "in copyDataTo SFCZVariable"));

	  SFCZVariableBase* v = oldDataWarehouse->d_sfczDB.get(currentVar.label_, currentVar.matlIndex_, currentVar.patch_);

	  IntVector newLowIndex = newPatch->getLowIndex(basis, currentVar.label_->getBoundaryLayer());
	  IntVector newHighIndex = newPatch->getHighIndex(basis, currentVar.label_->getBoundaryLayer());
	  //cout << "  newPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  IntVector copyLowIndex = Max(newLowIndex, lowIndex);
	  IntVector copyHighIndex = Min(newHighIndex, highIndex);
	  //cout << "  copPatch(" << newPatch->getID() << ") = " << newLowIndex << " to " << newHighIndex << endl;
	  //cout << "Var Low  = " << v->getLow() << endl;
	  //cout << "Var High = " << v->getHigh() << endl;
	  if ( !newDataWarehouse->d_sfczDB.exists(currentVar.label_, currentVar.matlIndex_, newPatch) ) {
	    SFCZVariableBase* newVariable = v->cloneType();
	    newVariable->rewindow( newLowIndex, newHighIndex );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	    newDataWarehouse->d_sfczDB.put(currentVar.label_, currentVar.matlIndex_, newPatch, newVariable, false);
	  } else {
	    SFCZVariableBase* newVariable = newDataWarehouse->d_sfczDB.get(currentVar.label_, currentVar.matlIndex_, newPatch );
	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
	  }
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH SFCZVARIABLE END" << endl;
	}
	break;
      case TypeDescription::ParticleVariable:
	{
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH PARTICLEVARIABLE BGN" << endl;
  	  if(!oldDataWarehouse->d_particleDB.exists(currentVar.label_, currentVar.matlIndex_, currentVar.patch_))
  	    SCI_THROW(UnknownVariable(currentVar.label_->getName(), oldDataWarehouse->getID(), currentVar.patch_, currentVar.matlIndex_,
  				      "in copyDataTo ParticleVariable"));
  	  if ( !newDataWarehouse->d_particleDB.exists(currentVar.label_, currentVar.matlIndex_, newPatch) ) {
            PatchSubset* ps = new PatchSubset;
            ps->add(currentVar.patch_);
            PatchSubset* newps = new PatchSubset;
            newps->add(newPatch);
            MaterialSubset* ms = new MaterialSubset;
            ms->add(currentVar.matlIndex_);
  	    newDataWarehouse->transferFrom(oldDataWarehouse, currentVar.label_, ps, ms, newps);
            delete ps;
            delete ms;
            delete newps;
  	  } else {
            SCI_THROW(InternalError("Particle copy not implemented for pre-existent var (BNR Regridder?)"));
  	  }
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH PARTICLEVARIABLE END" << endl;
          }
	break;
      case TypeDescription::PerPatch:
	{
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH PERPATCHVARIABLE BGN" << endl;
//  	  if(!oldDataWarehouse->d_perpatchDB.exists(currentVar.label_, currentVar.matlIndex_, currentVar.patch_))
//  	    SCI_THROW(UnknownVariable(currentVar.label_->getName(), oldDataWarehouse->getID(), currentVar.patch_, currentVar.matlIndex_,
//  				      "in copyDataTo PerPatch"));
//  	  PerPatchBase* v = oldDataWarehouse->d_perpatchDB.get(currentVar.label_, currentVar.matlIndex_, currentVar.patch_);
//  	  if ( !newDataWarehouse->d_perpatchDB.exists(currentVar.label_, currentVar.matlIndex_, newPatch) ) {
//  	    PerPatchBase* newVariable = v->cloneType();
//  	    newVariable->rewindow( newLowIndex, newHighIndex );
//  	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
//  	    newDataWarehouse->d_perpatchDB.put(currentVar.label_, currentVar.matlIndex_, newPatch, newVariable, false);
//  	  } else {
//  	    PerPatchBase* newVariable = newDataWarehouse->d_perpatchDB.get(currentVar.label_, currentVar.matlIndex_, newPatch );
//  	    newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
//  	  }
	  //cerr << getpid() << ": RANDY: SchedulerCommon::copyDataToNewGrid() FOR NEW PATCH PERPATCHVARIABLE END" << endl;
	}
	break;
      default:
	SCI_THROW(InternalError("Unknown variable type in transferFrom: "+currentVar.label_->getName()));
      }

    } // for ( int newPatchIndex = 0; newPatchIndex < neighbors.size(); newPatchIndex++) {

  } // for ( unsigned int i = 0; i < variableInfo.size(); i++ ) {

  // d_lock.writeUnlock(); Do we need this?

  dbg << "SchedulerCommon::copyDataToNewGrid() END" << endl;
}
