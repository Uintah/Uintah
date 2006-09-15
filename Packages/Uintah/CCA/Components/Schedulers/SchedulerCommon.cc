#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/LocallyComputedPatchVarMap.h>
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

#ifdef _WIN32
#include <time.h>
#endif

using namespace Uintah;
using namespace SCIRun;
using namespace std;

#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#else
#define SCISHARE
#endif
// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCISHARE SCIRun::Mutex       cerrLock;
extern DebugStream mixedDebug;

static DebugStream dbg("SchedulerCommon", false);

SchedulerCommon::SchedulerCommon(const ProcessorGroup* myworld, Output* oport)
  : UintahParallelComponent(myworld), m_outPort(oport),
    m_graphDoc(NULL), m_nodes(NULL)
{
  d_generation = 0;
  numOldDWs = 0;

  emit_taskgraph = false;
  memlogfile = 0;
  restartable = false;
  for(int i=0;i<Task::TotalDWs;i++)
    dwmap[i]=Task::InvalidDW;
  // Default mapping...
  dwmap[Task::OldDW]=0;
  dwmap[Task::NewDW]=1;

  m_locallyComputedPatchVarMap = scinew LocallyComputedPatchVarMap;
  reloc_new_posLabel_ = 0;
}

SchedulerCommon::~SchedulerCommon()
{
  if(memlogfile)
    delete memlogfile;

  // list of vars used for AMR regridding
  for (unsigned i = 0; i < label_matls_.size(); i++)
    for ( label_matl_map::iterator iter = label_matls_[i].begin(); iter != label_matls_[i].end(); iter++)
      if (iter->second->removeReference())
        delete iter->second;
  
  for (unsigned i = 0; i < graphs.size(); i++) {
    delete graphs[i];
  }

  label_matls_.clear();

  delete m_locallyComputedPatchVarMap;
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
  
  for (unsigned i = 0; i < graphs.size(); i++) {
    DetailedTasks* dts = graphs[i]->getDetailedTasks();
    if (dts) {
      dts->emitEdges(edgesElement, rank);
    }
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
SchedulerCommon::problemSetup(const ProblemSpecP& prob_spec,
                              SimulationStateP& state)
{
  d_sharedState = state;

  // Initializing trackingStartTime_ and trackingEndTime_ to default values
  // so that we do not crash when running MALLOC_STRICT.
  trackingStartTime_ = 1;
  trackingEndTime_ = 0;
  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if(params){
    ProblemSpecP track = params->findBlock("VarTracker");
    if (track) {
      track->require("start_time", trackingStartTime_);
      track->require("end_time", trackingEndTime_);
      track->getWithDefault("level", trackingLevel_, -1);
      track->getWithDefault("start_index", trackingStartIndex_, IntVector(-9,-9,-9));
      track->getWithDefault("end_index", trackingEndIndex_, IntVector(-9,-9,-9));

      for (ProblemSpecP var=track->findBlock("var"); var != 0; var = var->findNextBlock("var")) {
        map<string,string> attributes;
        var->getAttributes(attributes);
        string name = attributes["label"];
        trackingVars_.push_back(name);
        string dw = attributes["dw"];
        if (dw == "OldDW") trackingDWs_.push_back(Task::OldDW);
        else if (dw == "NewDW") trackingDWs_.push_back(Task::NewDW);
        else if (dw == "CoarseNewDW") trackingDWs_.push_back(Task::CoarseNewDW);
        else if (dw == "CoarseOldDW") trackingDWs_.push_back(Task::CoarseOldDW);
        else if (dw == "ParentOldDW") trackingDWs_.push_back(Task::ParentOldDW);
        else if (dw == "ParentOldDW") trackingDWs_.push_back(Task::ParentNewDW);
        else trackingDWs_.push_back(Task::NewDW);
      }
      for (ProblemSpecP task=track->findBlock("task"); task != 0; task = task->findNextBlock("task")) {
        map<string,string> attributes;
        task->getAttributes(attributes);
        string name = attributes["name"];
        trackingTasks_.push_back(name);
      }      
    }
  }
  noScrubVars_.insert("refineFlag");
  noScrubVars_.insert("refinePatchFlag");
}

void
SchedulerCommon::printTrackedVars(DetailedTask* dt, bool before)
{
  bool printedHeader = false;
  LoadBalancer* lb = getLoadBalancer();
 
  unsigned t;
  for (t = 0; t < trackingTasks_.size(); t++) {
    if (trackingTasks_[t] == dt->getTask()->getName())
      break;
  }

  // print for all tasks unless one is specified (but disclude DataArchiver tasks)
  if ((t == trackingTasks_.size() && trackingTasks_.size() != 0) || 
      ((string(dt->getTask()->getName())).substr(0,12) == "DataArchiver"))
    return;

  if (d_sharedState && (trackingStartTime_ > d_sharedState->getElapsedTime() || trackingEndTime_ < d_sharedState->getElapsedTime()))
    return;

  for (int i = 0; i < (int) trackingVars_.size(); i++) {
    bool printedVarName = false;

    // that DW may not have been mapped....
    if (dt->getTask()->mapDataWarehouse(trackingDWs_[i]) < 0 || 
        dt->getTask()->mapDataWarehouse(trackingDWs_[i]) >= (int) dws.size())
      continue;

    OnDemandDataWarehouseP dw = dws[dt->getTask()->mapDataWarehouse(trackingDWs_[i])];
    
    if (dw == 0) // old on initialization timestep
      continue;

    // get the level here, as the grid can be different between the old and new DW
    
    const Grid* grid = dw->getGrid();

    int levelnum;
    
    if (trackingLevel_ == -1) {
      levelnum = grid->numLevels() - 1;
    }
    else {
      levelnum = trackingLevel_;
      if (levelnum >= grid->numLevels())
        continue;
    }
    const LevelP level = grid->getLevel(levelnum);
    const VarLabel* label = VarLabel::find(trackingVars_[i]);

    cout.precision(16);

    if (!label)
      continue;

    const PatchSubset* patches = dt->getPatches();
    
    // a once-per-proc task is liable to have multiple levels, and thus calls to getLevel(patches) will fail
    if (dt->getTask()->getType() != Task::OncePerProc && (!patches || getLevel(patches)->getIndex() != levelnum))
      continue;
    for (int p = 0; patches && p < patches->size(); p++) {

      const Patch* patch = patches->get(p);

      // don't print ghost patches (dw->get will yell at you)
      if ((trackingDWs_[i] == Task::OldDW && lb->getOldProcessorAssignment(0,patch,0) != d_myworld->myrank()) ||
          (trackingDWs_[i] == Task::NewDW && lb->getPatchwiseProcessorAssignment(patch) != d_myworld->myrank()))
        continue;
      
      const TypeDescription* td = label->typeDescription();
      Patch::VariableBasis basis = patch->translateTypeToBasis(td->getType(), false);
      IntVector start = 
          Max(patch->getLowIndex(basis, IntVector(0,0,0)), trackingStartIndex_);
      IntVector end = 
        Min(patch->getHighIndex(basis, IntVector(0,0,0)), trackingEndIndex_);

      // loop over matls too
      for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
        if (!dw->exists(label, m, patch))
          continue;
        if (!(start.x() < end.x() && start.y() < end.y() && start.z() < end.z()))
          continue;        
        if (td->getSubType()->getType() != TypeDescription::double_type &&
            td->getSubType()->getType() != TypeDescription::Vector)
          // only allow *Variable<double> and *Variable<Vector> for now
          continue;

        // pending the task that allocates the var, we may not have allocated it yet
        GridVariable* v;
        switch (td->getType()) {
        case TypeDescription::CCVariable:
        case TypeDescription::NCVariable:
        case TypeDescription::SFCXVariable:
        case TypeDescription::SFCYVariable:
        case TypeDescription::SFCZVariable: 
          v = dynamic_cast<GridVariable*>(dw->d_varDB.get(label, m, patch));
          break;
        default: 
          throw InternalError("Cannot track var type of non-grid-type",__FILE__,__LINE__); break;
        }
        start = Max(start, v->getLow());
        end = Min(end, v->getHigh());
        if (!(start.x() < end.x() && start.y() < end.y() && start.z() < end.z())) 
          continue;

        if (!printedHeader) {
          cout << d_myworld->myrank() << (before ? " BEFORE" : " AFTER") << " execution of " 
               << *dt << endl;
          printedHeader = true;
        }
        if (!printedVarName) {
          cout << d_myworld->myrank() << "  Variable: " << trackingVars_[i] << ", DW " << dw->getID() << ", Patch " << patch->getID() << ", Matl " << m << endl;
          if (trackingVars_[i] == "rho_CC")
            cout << "  RHO: " << dw->getID() << " original input " << trackingDWs_[i] << endl;
        }
            
        // now get it the way a normal task would get it
        switch (td->getSubType()->getType()) {
        case TypeDescription::double_type: {
          if (td->getType() == TypeDescription::CCVariable) {
            constCCVariable<double> var;
            dw->get(var, label, m, patch, Ghost::None, 0);
            
            for (int z = start.z(); z < end.z(); z++) {
              for (int y = start.y(); y < end.y(); y++) {
                cout << d_myworld->myrank() << "  ";
                for (int x = start.x(); x < end.x(); x++) {
                  IntVector c(x,y,z);
                  cout << " " << c << ": " << var[c];
                }
                cout << endl;
              }
              cout << endl;
            }
          }
          else if (td->getType() == TypeDescription::NCVariable) {
            constNCVariable<double> var;
            dw->get(var, label, m, patch, Ghost::None, 0);
            
            for (int z = start.z(); z < end.z(); z++) {
              for (int y = start.y(); y < end.y(); y++) {
                cout << d_myworld->myrank() << "  ";
                for (int x = start.x(); x < end.x(); x++) {
                  IntVector c(x,y,z);
                  cout << " " << c << ": " << var[c];
                }
                cout << endl;
              }
              cout << endl;
            }
          }
          
          else if (td->getType() == TypeDescription::SFCXVariable) {
            constSFCXVariable<double> var;
            dw->get(var, label, m, patch, Ghost::None, 0);
            
            for (int z = start.z(); z < end.z(); z++) {
              for (int y = start.y(); y < end.y(); y++) {
                cout << d_myworld->myrank() << "  ";
                for (int x = start.x(); x < end.x(); x++) {
                  IntVector c(x,y,z);
                  cout << " " << c << ": " << var[c];
                }
                cout << endl;
              }
              cout << endl;
            }
          }
          else if (td->getType() == TypeDescription::SFCYVariable) {
            constSFCYVariable<double> var;
            dw->get(var, label, m, patch, Ghost::None, 0);
            
            for (int z = start.z(); z < end.z(); z++) {
              for (int y = start.y(); y < end.y(); y++) {
                cout << d_myworld->myrank() << "  ";
                for (int x = start.x(); x < end.x(); x++) {
                  IntVector c(x,y,z);
                  cout << " " << c << ": " << var[c];
                }
                cout << endl;
              }
              cout << endl;
            }
          }
          else if (td->getType() == TypeDescription::SFCZVariable) {
            constSFCZVariable<double> var;
            dw->get(var, label, m, patch, Ghost::None, 0);
            
            for (int z = start.z(); z < end.z(); z++) {
              for (int y = start.y(); y < end.y(); y++) {
                cout << d_myworld->myrank() << "  ";
                for (int x = start.x(); x < end.x(); x++) {
                  IntVector c(x,y,z);
                  cout << " " << c << ": " << var[c];
                }
                cout << endl;
              }
              cout << endl;
            }
          }
        }
          break;
        case TypeDescription::Vector: {
          constCCVariable<Vector> var;
          dw->get(var, label, m, patch, Ghost::None, 0);
          
          for (int z = start.z(); z < end.z(); z++) {
            for (int y = start.y(); y < end.y(); y++) {
              cout << d_myworld->myrank() << "  ";
              for (int x = start.x(); x < end.x(); x++) {
                IntVector c(x,y,z);
                cout << " " << c << ": " << var[c];
              }
              cout << endl;
            }
            cout << endl;
          }
        }
          break;
        default: break;
        }
      }
    }
  }
}

LoadBalancer*
SchedulerCommon::getLoadBalancer()
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   return lb;
}

void
SchedulerCommon::addTaskGraph(Scheduler::tgType type)
{
  TaskGraph* tg = scinew TaskGraph(this, d_myworld, type);
  tg->initialize();
  graphs.push_back(tg);
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

  graphs[graphs.size()-1]->addTask(task, patches, matls);
  numTasks_++;

  for (Task::Dependency* dep = task->getRequires(); dep != 0;
       dep = dep->next) {
    if(isOldDW(dep->mapDataWarehouse())) {
      d_initRequires.push_back(dep);
      d_initRequiredVars.insert(dep->var);
    }
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
SchedulerCommon::initialize(int numOldDW /* =1 */, int numNewDW /* =1 */)
{

  // doesn't really do anything except initialize/clear the taskgraph
  //   if the default parameter values are used
  int numDW = numOldDW+numNewDW;
  int oldnum = (int)dws.size();

  // in AMR cases we will often need to move from many new DWs to one.  In those cases, move the last NewDW to be the next new one.
  if (oldnum - numOldDWs > 1) {
    dws[numDW-1] = dws[oldnum-1];
  }

  // Clear out the data warehouse so that memory will be freed
  for(int i=numDW;i<oldnum;i++)
    dws[i]=0;
  dws.resize(numDW);
  for(;oldnum < numDW; oldnum++)
    dws[oldnum] = 0;
  numOldDWs = numOldDW;

  // clear the taskgraphs, and set the first one
  for (unsigned i = 0; i < graphs.size(); i++) {
    delete graphs[i];
  }

  graphs.clear();

  d_initRequires.clear();
  d_initRequiredVars.clear();
  numTasks_ = 0;

  addTaskGraph(NormalTaskGraph);

  m_ghostOffsetVarMap.clear();

}

void SchedulerCommon::setParentDWs(DataWarehouse* parent_old_dw, DataWarehouse* parent_new_dw)
{
  OnDemandDataWarehouse* pold = dynamic_cast<OnDemandDataWarehouse*>(parent_old_dw);
  OnDemandDataWarehouse* pnew = dynamic_cast<OnDemandDataWarehouse*>(parent_new_dw);
  if(parent_old_dw && parent_new_dw){
    ASSERT(pold != 0);
    ASSERT(pnew != 0);
    ASSERT(numOldDWs > 2);
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
    m_locallyComputedPatchVarMap->getConnectedPatchGroup(patch);
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
    VarLabelMatl<Patch> vmp(label, matlIndex, memberPatch);
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
  
  for (unsigned i = 0; i < graphs.size(); i++) {
    DetailedTasks* dts = graphs[i]->getDetailedTasks();
    if (dts) {
      dts->logMemoryUse(*memlogfile, total, "Taskgraph");
    }
  }
  *memlogfile << "Total: " << total << '\n';
  memlogfile->flush();
}

// Makes and returns a map that maps strings to VarLabels of
// that name and a list of material indices for which that
// variable is valid (according to d_allcomps in graph).
Scheduler::VarLabelMaterialMap* SchedulerCommon::makeVarLabelMaterialMap()
{
  VarLabelMaterialMap* result = scinew VarLabelMaterialMap;
  for (unsigned i = 0; i < graphs.size(); i++) {
    graphs[i]->makeVarLabelMaterialMap(result);
  }
  return result;
}
     
void SchedulerCommon::doEmitTaskGraphDocs()
{
  emit_taskgraph=true;
}

void SchedulerCommon::compile()
{
  GridP grid = const_cast<Grid*>(getLastDW()->getGrid());
  GridP oldGrid;
  if (dws[0])
    oldGrid = const_cast<Grid*>(get_dw(0)->getGrid());
  if(numTasks_ > 0){
    TAU_PROFILE("SchedulerCommon::compile()", " ", TAU_USER); 

    dbg << d_myworld->myrank() << " SchedulerCommon starting compile\n";
    
    // pass the first to the rest, so we can share the scrubcountTable
    DetailedTasks* first = 0;
    for (unsigned i = 0; i < graphs.size(); i++) {
      if (graphs.size() > 1)
        dbg << d_myworld->myrank() << "  Compiling graph#" << i << " of " << graphs.size() << endl;
      DetailedTasks* dts = graphs[i]->createDetailedTasks(useInternalDeps(), first, grid, oldGrid);
      if (!first)
        first = dts;
    }
    verifyChecksum();
    dbg << d_myworld->myrank() << " SchedulerCommon finished compile\n";
  }

  m_locallyComputedPatchVarMap->reset();

#if 1
  for (int i = 0; i < grid->numLevels(); i++) {
    const PatchSubset* patches = getLoadBalancer()->getPerProcessorPatchSet(grid->getLevel(i))->getSubset(d_myworld->myrank());
    if (patches->size() > 0)
      m_locallyComputedPatchVarMap->addComputedPatchSet(patches);
  }
#else
  for (unsigned i = 0; i < graphs.size(); i++) { 
    DetailedTasks* dts = graphs[i]->getDetailedTasks();
    
    if (dts != 0) {    
      // figure out the locally computed patches for each variable.
      for (int i = 0; i < dts->numLocalTasks(); i++) {
        const DetailedTask* dt = dts->localTask(i);
        for(const Task::Dependency* comp = dt->getTask()->getComputes();
            comp != 0; comp = comp->next){
          if (comp->var->typeDescription()->getType() != TypeDescription::ReductionVariable) {
            constHandle<PatchSubset> patches =
              comp->getPatchesUnderDomain(dt->getPatches());
            m_locallyComputedPatchVarMap->addComputedPatchSet(patches.get_rep());
          }
        }
      }
    }
  }
#endif
  m_locallyComputedPatchVarMap->makeGroups();
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
SchedulerCommon::scheduleAndDoDataCopy(const GridP& grid, SimulationInterface* sim)
{  
  // TODO - use the current initReqs and push them back, instead of doing this...
  // clear the old list of vars and matls
  for (unsigned i = 0; i < label_matls_.size(); i++)
    for ( label_matl_map::iterator iter = label_matls_[i].begin(); iter != label_matls_[i].end(); iter++)
      if (iter->second->removeReference())
        delete iter->second;
  
  label_matls_.clear();
  label_matls_.resize(grid->numLevels());

  // produce a map from all tasks' requires from the Old DW.  Store the varlabel and matls
  // TODO - only do this ONCE.
  for (unsigned t = 0; t < graphs.size(); t++) {
    TaskGraph* tg = graphs[t];
    for (int i = 0; i < tg->getNumTasks(); i++) {
      Task* task = tg->getTask(i);
      if (task->getType() == Task::Output)
        continue;  
      for(Task::Dependency* dep = task->getRequires(); dep != 0; dep=dep->next){
        bool copyThisVar = dep->whichdw == Task::OldDW;
        // override to manually copy a var
        if (!copyThisVar)
          if (copyDataVars_.find(dep->var->getName()) != copyDataVars_.end())
            copyThisVar = true;

        if (copyThisVar) {
          if (dep->var->typeDescription()->getType() == TypeDescription::ReductionVariable)
            // we will take care of reduction variables in a different section
            continue;
          
          // check the level on the case where variables are only computed on certain levels
          const PatchSet* ps = task->getPatchSet();
          int level = -1;
          if (dep->patches) // just in case the task is over multiple levels...
            level = getLevel(dep->patches)->getIndex();
          else if (ps)
            level = getLevel(ps)->getIndex();
          
          // we don't want data with an invalid level, or requiring from a different level (remember, we are
          // using an old task graph).  That willbe copied later (and chances are, it's to modify anyway).
          if (level == -1 || level > grid->numLevels()-1 || dep->patches_dom == Task::CoarseLevel || 
              dep->patches_dom == Task::FineLevel)
            continue;
          
          const MaterialSubset* matSubset = (dep->matls != 0) ?
            dep->matls : dep->task->getMaterialSet()->getUnion();
          
          
          // if var was already found, make a union of the materials
          MaterialSubset* matls = scinew MaterialSubset(matSubset->getVector());
          matls->addReference();
          
          MaterialSubset* union_matls;
          union_matls = label_matls_[level][dep->var];
          
          if (union_matls) {
            for (int i = 0; i < union_matls->size(); i++) {
              if (!matls->contains(union_matls->get(i))) {
                matls->add(union_matls->get(i)); 
              } 
            }
            if (union_matls->removeReference()) {
              delete union_matls;
            }
          }
          matls->sort();
          label_matls_[level][dep->var] = matls;
        }
      }
    }
  }

  this->initialize(1, 1);
  this->advanceDataWarehouse(grid);
  this->clearMappings();
  this->mapDataWarehouse(Task::OldDW, 0);
  this->mapDataWarehouse(Task::NewDW, 1);
  this->mapDataWarehouse(Task::CoarseOldDW, 0);
  this->mapDataWarehouse(Task::CoarseNewDW, 1);
  
  DataWarehouse* oldDataWarehouse = this->get_dw(0);


  DataWarehouse* newDataWarehouse = this->getLastDW();

  oldDataWarehouse->setScrubbing(DataWarehouse::ScrubNone);
  newDataWarehouse->setScrubbing(DataWarehouse::ScrubNone);
  const Grid* oldGrid = oldDataWarehouse->getGrid();
  vector<Task*> dataTasks;
  vector<Handle<PatchSet> > refineSets(grid->numLevels(),(PatchSet*)0);
  SchedulerP sched(dynamic_cast<Scheduler*>(this));

  d_sharedState->setCopyDataTimestep(true);

  for (int i = 0; i < grid->numLevels(); i++) {
    LevelP newLevel = newDataWarehouse->getGrid()->getLevel(i);

    if (i > 0) {
      if (i >= oldGrid->numLevels()) {
        // new level - refine everywhere
        refineSets[i] = const_cast<PatchSet*>(newLevel->eachPatch());
      }
      // find patches with new space - but temporarily, refine everywhere... 
      else if (i < oldGrid->numLevels()) {
        refineSets[i] = scinew PatchSet;
        LevelP oldLevel = oldDataWarehouse->getGrid()->getLevel(newLevel->getIndex());
        
        // go through the patches, and find if there are patches that weren't entirely 
        // covered by patches on the old grid, and interpolate them.  
        // then after, copy the data, and if necessary, overwrite interpolated data
        
        for (Level::patchIterator iter = newLevel->patchesBegin(); iter != newLevel->patchesEnd(); iter++) {
          Patch* newPatch = *iter;
          
          // get the low/high for what we'll need to get
          IntVector lowIndex, highIndex;
          //newPatch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0), Ghost::None, 0, lowIndex, highIndex);
          lowIndex = newPatch->getInteriorCellLowIndex();
          highIndex = newPatch->getInteriorCellHighIndex();
          
          // find if area on the new patch was not covered by the old patches
          IntVector dist = highIndex-lowIndex;
          int totalCells = dist.x()*dist.y()*dist.z();
          int sum = 0;
          Patch::selectType oldPatches;
          oldLevel->selectPatches(lowIndex, highIndex, oldPatches);
          
          for (int old = 0; old < oldPatches.size(); old++) {
            const Patch* oldPatch = oldPatches[old];
            IntVector oldLow = oldPatch->getInteriorCellLowIndex();
            IntVector oldHigh = oldPatch->getInteriorCellHighIndex();

            IntVector low = Max(oldLow, lowIndex);
            IntVector high = Min(oldHigh, highIndex);
            IntVector dist = high-low;
            sum += dist.x()*dist.y()*dist.z();
          }  // for oldPatches
          if (sum != totalCells) {
            refineSets[i]->add(newPatch);
          }
          
        } // for patchIterator
      }
      if (refineSets[i]->size() > 0) {
        dbg << d_myworld->myrank() << "  Calling scheduleRefine for patches " << *refineSets[i].get_rep() << endl;
        sim->scheduleRefine(refineSets[i].get_rep(), sched);
      }
    }

    // find the patches that you don't refine
    Handle<PatchSubset> temp = scinew PatchSubset; // temp only to show empty set.  Don't pass into computes
    constHandle<PatchSubset> modset, levelset, compset, diffset, intersection;
    
    if (refineSets[i])
      modset = refineSets[i]->getUnion();
    else {
      modset = temp;
    }
    levelset = newLevel->eachPatch()->getUnion();
    
    PatchSubset::intersectionAndDifferences(levelset, modset, intersection, compset, diffset);

    dataTasks.push_back(scinew Task("SchedulerCommon::copyDataToNewGrid", this,                          
                                     &SchedulerCommon::copyDataToNewGrid));
    for ( label_matl_map::iterator iter = label_matls_[i].begin(); iter != label_matls_[i].end(); iter++) {
      const VarLabel* var = iter->first;
      MaterialSubset* matls = iter->second;

      dataTasks[i]->requires(Task::OldDW, var, 0, Task::OtherGridDomain, matls, Task::NormalDomain, Ghost::None, 0);
      if (compset && compset->size() > 0) {
        dbg << "  Scheduling copy for var " << *var << " matl " << *matls << " Computes: " << *compset.get_rep() << endl;
        dataTasks[i]->computes(var, compset.get_rep(), matls);
      }
      if (modset && modset->size() > 0) {
        dbg << "  Scheduling copy for var " << *var << " matl " << " Modifies: " << *modset.get_rep() << endl;
        dataTasks[i]->modifies(var, modset.get_rep(), matls);
      }
    }
    addTask(dataTasks[i], newLevel->eachPatch(), d_sharedState->allMaterials());
    if (i > 0) {
      sim->scheduleRefineInterface(newLevel, sched, 0, 1);
    }
  }

  // set so the load balancer will make an adequate neighborhood, as the default
  // neighborhood isn't good enough for the copy data timestep

#ifndef _WIN32
  const char* tag = AllocatorSetDefaultTag("DoDataCopy");
#endif
  this->compile(); 
  this->execute();
#ifndef _WIN32
  AllocatorSetDefaultTag(tag);
#endif

  d_sharedState->setCopyDataTimestep(false);

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
    delete v; // copied on the put command
  }
  newDataWarehouse->refinalize();
}


void
SchedulerCommon::copyDataToNewGrid(const ProcessorGroup*, const PatchSubset* patches,
                                   const MaterialSubset* matls, DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  dbg << "SchedulerCommon::copyDataToNewGrid() BGN on patches " << *patches  << endl;

  OnDemandDataWarehouse* oldDataWarehouse = dynamic_cast<OnDemandDataWarehouse*>(old_dw);
  OnDemandDataWarehouse* newDataWarehouse = dynamic_cast<OnDemandDataWarehouse*>(new_dw);

  // For each patch in the patch subset which contains patches in the new grid
  for ( int p = 0; p < patches->size(); p++ ) {
    const Patch* newPatch = patches->get(p);
    const Level* newLevel = newPatch->getLevel();

    // to create once per matl instead of once per matl-var
    vector<ParticleSubset*> oldsubsets(d_sharedState->getNumMatls()), newsubsets(d_sharedState->getNumMatls());

    // If there is a level that didn't exist, we don't need to copy it
    if ( newLevel->getIndex() >= oldDataWarehouse->getGrid()->numLevels() ) {
      continue;
    }
    
    // find old patches associated with this patch
    LevelP oldLevel = oldDataWarehouse->getGrid()->getLevel( newLevel->getIndex() );
    
    for ( label_matl_map::iterator iter = label_matls_[oldLevel->getIndex()].begin(); 
          iter != label_matls_[oldLevel->getIndex()].end(); iter++) {
      const VarLabel* label = iter->first;
      MaterialSubset* var_matls = iter->second;


      // get the low/high for what we'll need to get
      Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), true);
      IntVector newLowIndex, newHighIndex;
      newPatch->computeVariableExtents(basis, IntVector(0,0,0), Ghost::None, 0, newLowIndex, newHighIndex);

      for (int m = 0; m < var_matls->size(); m++) {
        int matl = var_matls->get(m);

        if (!matls->contains(matl)) {
          //cout << "We are skipping material " << currentVar.matlIndex_ << endl;
          continue;
        }

        if (label->typeDescription()->getType() != TypeDescription::ParticleVariable) {
          Patch::selectType oldPatches;
          oldLevel->selectPatches(newLowIndex, newHighIndex, oldPatches);
          
          for ( int oldIdx = 0;  oldIdx < oldPatches.size(); oldIdx++) {
            const Patch* oldPatch = oldPatches[oldIdx];
            
            if(!oldDataWarehouse->exists(label, matl, oldPatch))
              continue; // see comment about oldPatchToTest in ScheduleAndDoDataCopy
            IntVector oldLowIndex;
            IntVector oldHighIndex;
            
            if (newLevel->getIndex() > 0) {
              oldLowIndex = oldPatch->getInteriorLowIndexWithBoundary(basis);
              oldHighIndex = oldPatch->getInteriorHighIndexWithBoundary(basis);
            }
            else {
              oldLowIndex = oldPatch->getLowIndex(basis, label->getBoundaryLayer());
              oldHighIndex = oldPatch->getHighIndex(basis, label->getBoundaryLayer());
            }
            
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
                                          "in copyDataTo GridVariable", __FILE__, __LINE__));
              GridVariable* v = dynamic_cast<GridVariable*>(oldDataWarehouse->d_varDB.get(label, matl, oldPatch));
              
              if ( !newDataWarehouse->exists(label, matl, newPatch) ) {
                GridVariable* newVariable = v->cloneType();
                newVariable->rewindow( newLowIndex, newHighIndex );
                newVariable->copyPatch( v, copyLowIndex, copyHighIndex );
                newDataWarehouse->d_varDB.put(label, matl, newPatch, newVariable, false);
              } else {
                GridVariable* newVariable = 
                  dynamic_cast<GridVariable*>(newDataWarehouse->d_varDB.get(label, matl, newPatch ));
                // make sure it exists in the right region (it might be ghost data)
                newVariable->rewindow(newLowIndex, newHighIndex);
                if (oldPatch->isVirtual()) {
                  // it can happen where the old patch was virtual and this is not
                  GridVariable* tmpVar = newVariable->cloneType();
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
            case TypeDescription::PerPatch:
              {
              }
              break;
            default:
              SCI_THROW(InternalError("Unknown variable type in transferFrom: "+label->getName(), __FILE__, __LINE__));
            } // end switch
          } // end oldPatches
        }
        else {
          ParticleSubset* oldsub = oldsubsets[matl];
          if (!oldsub) {
            // collect the particles from the range encompassing this patch.  Use interior cells since
            // extracells aren't collected across processors in the data copy, and they don't matter
            // for particles anyhow (but we will have to reset the bounds to copy the data)
            oldsub = oldDataWarehouse->getParticleSubset(matl, newPatch->getInteriorLowIndexWithBoundary(Patch::CellBased),
                                                         newPatch->getInteriorHighIndexWithBoundary(Patch::CellBased), 
                                                         oldLevel.get_rep(), 0, reloc_new_posLabel_);
            oldsubsets[matl] = oldsub;
            oldsub->addReference();
          }

          ParticleSubset* newsub = newsubsets[matl];
          // it might have been created in Refine
          if (!newsub) {
            if (!newDataWarehouse->haveParticleSubset(matl, newPatch))
              newsub = newDataWarehouse->createParticleSubset(oldsub->numParticles(), matl, newPatch);
            else {
              newsub = newDataWarehouse->getParticleSubset(matl, newPatch);
              ASSERT(newsub->numParticles() == 0);
              newsub->addParticles(oldsub->numParticles());
            }
            newsubsets[matl] = newsub;
          }

          ParticleVariableBase* newv = dynamic_cast<ParticleVariableBase*>(label->typeDescription()->createInstance());
          newv->allocate(newsub);
          // don't get and copy if there were no old patches
          if (oldsub->getNeighbors().size() > 0) {
            
            constParticleVariableBase* var = newv->cloneConstType();
            oldDataWarehouse->get(*var, label, oldsub);
            
            // reset the bounds of the old var's data so copyData doesn't complain
            ParticleSet* pset = scinew ParticleSet(oldsub->numParticles());
            ParticleSubset* tempset = scinew ParticleSubset(pset, true, matl, newPatch, newPatch->getLowIndex(), 
                                                            newPatch->getHighIndex(), 0);
            const_cast<ParticleVariableBase*>(&var->getBaseRep())->setParticleSubset(tempset);
            newv->copyData(&var->getBaseRep());
            delete var; //pset and tempset are deleted with it.
          }
          newDataWarehouse->put(*newv, label, true);
        }
      } // end matls
    } // end label_matls
    for (unsigned i = 0; i < oldsubsets.size(); i++)
      if (oldsubsets[i] && oldsubsets[i]->removeReference())
        delete oldsubsets[i];
  } // end patches

  // d_lock.writeUnlock(); Do we need this?

  dbg << "SchedulerCommon::copyDataToNewGrid() END" << endl;
}

void SchedulerCommon::scheduleDataCopyVar(string var)
{
  copyDataVars_.insert(var);
  noScrubVars_.insert(var);
}

void SchedulerCommon::scheduleNoScrubVar(string var)
{
  noScrubVars_.insert(var);
}

