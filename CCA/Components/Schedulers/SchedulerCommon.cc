
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
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

SchedulerCommon::SchedulerCommon(const ProcessorGroup* myworld, Output* oport)
  : UintahParallelComponent(myworld), m_outPort(oport), m_graphDoc(NULL),
    m_nodes(NULL)
{
  dws_[ Task::OldDW ] = dws_[ Task::NewDW ] = 0;
  dts_ = 0;
  emit_taskgraph = false;
  memlogfile = 0;
}

SchedulerCommon::~SchedulerCommon()
{
  dws_[ Task::OldDW ] = dws_[ Task::NewDW ] = 0;
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
  
  DOM_DOMImplementation impl;
  m_graphDoc = scinew DOM_Document();
  *m_graphDoc = impl.createDocument(0, "Uintah_TaskGraph", DOM_DocumentType());
  DOM_Element root = m_graphDoc->getDocumentElement();
  
  DOM_Element meta = m_graphDoc->createElement("Meta");
  root.appendChild(meta);
  appendElement(meta, "username", getenv("LOGNAME"));
  time_t t = time(NULL);
  appendElement(meta, "date", ctime(&t));
  
  m_nodes = scinew DOM_Element(m_graphDoc->createElement("Nodes"));
  root.appendChild(*m_nodes);
  
  DOM_Element edgesElement = m_graphDoc->createElement("Edges");
  root.appendChild(edgesElement);
  
  if (dts_) {
    dts_->emitEdges(edgesElement, rank);
  }
  
  if (m_outPort->wasOutputTimestep()) {
    string timestep_dir(m_outPort->getLastTimestepOutputLocation());
    
    ostringstream fname;
    fname << "/taskgraph_" << setw(5) << setfill('0') << rank << ".xml";
    string file_name(timestep_dir + fname.str());
    ofstream graphfile(file_name.c_str());
    if (!graphfile) {
      cerr << "SchedulerCommon::emitEdges(): unable to open output file!\n";
      return;	// dependency dump failure shouldn't be fatal to anything else
    }
    graphfile << *m_graphDoc << endl;
  }
}

bool
SchedulerCommon::useInternalDeps()
{
  // keep track of internal dependencies only if it will emit
  // the taskgraphs (by default).
  return emit_taskgraph;
}

void
SchedulerCommon::emitNode(const DetailedTask* task, double start, double duration, double execution_duration, long long execution_flops, long long communication_flops)
{  
    if (m_nodes == NULL)
    	return;
    
    DOM_Element node = m_graphDoc->createElement("node");
    m_nodes->appendChild(node);

    appendElement(node, "name", task->getName());
    appendElement(node, "start", start);
    appendElement(node, "duration", duration);
    if (execution_duration > 0)
      appendElement(node, "execution_duration", execution_duration);
    if (execution_flops > 0)
      appendElement(node, "execution_flops", (long)execution_flops);
    if (communication_flops > 0)
      appendElement(node, "communication_flops", (long)communication_flops);
}

void
SchedulerCommon::finalizeNodes(int process /* = 0*/)
{
    if (m_graphDoc == NULL)
    	return;

    if (m_outPort->wasOutputTimestep()) {
      string timestep_dir(m_outPort->getLastTimestepOutputLocation());
      
      ostringstream fname;
      fname << "/taskgraph_" << setw(5) << setfill('0') << process << ".xml";
      string file_name(timestep_dir + fname.str());
      ofstream graphfile(file_name.c_str());
      graphfile << *m_graphDoc << endl;
    }
    
    delete m_nodes;
    m_nodes = NULL;
    delete m_graphDoc;
    m_graphDoc = NULL;
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
   graph.addTask(task, patches, matls);
   for (Task::Dependency* dep = task->getRequires(); dep != 0;
	dep = dep->next) {
     // Store the ghost cell information of each of the requires
     // so we can predict the total allocation needed for each variable.
     if (dep->numGhostCells > 0) {
       const PatchSubset* dep_patches;
       const MaterialSubset* dep_matls;
       if (dep->patches_dom == Task::NormalDomain)
	 dep_patches = patches->getUnion();
       else if (dep->patches_dom == Task::OutOfDomain)
	 dep_patches = dep->patches;
       else {
	 throw InternalError("Unhandled patches_dom with > 0 ghost cells");
       }
       if (dep->matls_dom == Task::NormalDomain)
	 dep_matls = matls->getUnion();
       else if (dep->matls_dom == Task::OutOfDomain)
	 dep_matls = dep->matls;
       else {
	 throw InternalError("Unhandled patches_dom with > 0 ghost cells");
       }
       m_ghostOffsetVarMap.includeOffsets(dep->var, dep_matls, dep_patches,
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
SchedulerCommon::initialize()
{
   graph.initialize();
   m_ghostOffsetVarMap.clear();
}

void 
SchedulerCommon::set_old_dw(DataWarehouse* oldDW)
{
  OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(oldDW);
  if (dw) {
    dws_[ Task::OldDW ] = dw;
  }
  else {
    throw InternalError("SchedulerCommon::set_old_dw: expecting OnDemandDataWarehous");
  }
}

void 
SchedulerCommon::set_new_dw(DataWarehouse* newDW)
{
  OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(newDW);
  if (dw) {
    dws_[ Task::NewDW ] = dw;
  }
  else {
    throw InternalError("SchedulerCommon::set_old_dw: expecting OnDemandDataWarehous");
  }
}

  void 
SchedulerCommon::advanceDataWarehouse(const GridP& grid)
{
  dws_[ Task::OldDW ] = dws_[ Task::NewDW ];
  int generation = d_generation++;
  if (dws_[Task::OldDW] == 0) {
    // first datawarehouse -- indicate that it is the "initialization"= dw.
    dws_[Task::NewDW] = scinew
      OnDemandDataWarehouse(d_myworld, this, generation, grid,
			    true /* initialization dw */);
  }
  else {
    dws_[Task::NewDW]=scinew OnDemandDataWarehouse(d_myworld, this, generation,
						   grid);
  }
}

DataWarehouse*
SchedulerCommon::get_old_dw()
{
  return dws_[Task::OldDW].get_rep();
}

DataWarehouse*
SchedulerCommon::get_new_dw()
{
  return dws_[ Task::NewDW ].get_rep();
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
  bool containsGivenPatch;
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
  if( dws_[ Task::OldDW ] )
    dws_[ Task::OldDW ]->logMemoryUse(*memlogfile, total, "OldDW");
  if( dws_[ Task::NewDW ] )
    dws_[ Task::NewDW ]->logMemoryUse(*memlogfile, total, "NewDW");
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

void SchedulerCommon::compile( const ProcessorGroup * pg, bool scrub_new)
{
  bool needScrubExtraneous = dts_ ? !dts_->doScrubNew() : false;
  
  actuallyCompile(pg, scrub_new);
  if (dts_ != 0) {
    dts_->computeLocalTasks(pg->myrank());
    
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

  if (needScrubExtraneous) {
    // The last one was compiled with no new scrubbing (i.e. initialization);
    // now scrub the OldDW data that won't be required by this new taskgraph.
    dts_->scrubExtraneousOldDW();
  }
}
