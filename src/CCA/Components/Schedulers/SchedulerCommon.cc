/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <TauProfilerForSCIRun.h>

#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SimulationInterface.h>

#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <string>
#include <vector>

#include <time.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream schedulercommon_dbg("SchedulerCommon_DBG", false);

// for calculating memory usage when sci-malloc is disabled.
char* SchedulerCommon::start_addr = NULL;


SchedulerCommon::SchedulerCommon(const ProcessorGroup* myworld,
                                 const Output*         oport)
  : UintahParallelComponent(myworld),
    m_outPort(oport),
    trackingVarsPrintLocation_(0),
    d_maxMemUse(0),
    m_graphDoc(NULL),
    m_nodes(NULL)
{
  d_generation = 0;
  numOldDWs    = 0;

  emit_taskgraph     = false;
  d_useSmallMessages = true;
  restartable        = false;
  memlogfile         = 0;

  for (int i = 0; i < Task::TotalDWs; i++) {
    dwmap[i] = Task::InvalidDW;
  }

  // Default mapping...
  dwmap[Task::OldDW] = 0;
  dwmap[Task::NewDW] = 1;

  d_isInitTimestep = false;
  d_isRestartInitTimestep = false;

  m_locallyComputedPatchVarMap = scinew LocallyComputedPatchVarMap;
  reloc_new_posLabel_ = 0;

  maxGhostCells.clear();
  maxLevelOffsets.clear();
}

//______________________________________________________________________
//
SchedulerCommon::~SchedulerCommon()
{
  if(memlogfile)
    delete memlogfile;

  // list of vars used for AMR regridding
  for (unsigned i = 0; i < label_matls_.size(); i++)
    for ( label_matl_map::iterator iter = label_matls_[i].begin(); iter != label_matls_[i].end(); iter++)
      if (iter->second->removeReference()) {
        delete iter->second;
      }
  
  for (unsigned i = 0; i < graphs.size(); i++) {
    delete graphs[i];
  }

  label_matls_.clear();

  delete m_locallyComputedPatchVarMap;
}

//______________________________________________________________________
//
void
SchedulerCommon::checkMemoryUse(unsigned long& memuse,
                                unsigned long& highwater,
                                unsigned long& maxMemUse)
{
  highwater = 0; 
  memuse    = 0;

#if !defined(DISABLE_SCI_MALLOC)
  size_t nalloc,  sizealloc, nfree,  sizefree, nfillbin,
         nmmap, sizemmap, nmunmap, sizemunmap, highwater_alloc,  
         highwater_mmap, bytes_overhead, bytes_free, bytes_fragmented, bytes_inuse, bytes_inhunks;
  
  GetGlobalStats( DefaultAllocator(),
                  nalloc, sizealloc, nfree, sizefree,
                  nfillbin, nmmap, sizemmap, nmunmap,
                  sizemunmap, highwater_alloc, highwater_mmap,
                  bytes_overhead, bytes_free, bytes_fragmented, bytes_inuse, bytes_inhunks );
  memuse = sizealloc - sizefree;
  highwater = highwater_mmap;

#else
  if ( ProcessInfo::isSupported( ProcessInfo::MEM_SIZE ) ) {
    memuse = ProcessInfo::getMemoryResident();
    //printf("1) memuse is %d (on proc %d)\n", (int)memuse, Uintah::Parallel::getMPIRank() );
  } else {
    memuse = (char*)sbrk(0)-start_addr;
    // printf("2) memuse is %d (on proc %d)\n", (int)memuse, Uintah::Parallel::getMPIRank() );
  }
#endif

  if( memuse > d_maxMemUse ) {
    // printf("Max memuse increased\n");
    d_maxMemUse = memuse;
  }
  maxMemUse = d_maxMemUse;
}

void
SchedulerCommon::resetMaxMemValue()
{
  d_maxMemUse = 0;
}

//______________________________________________________________________
//
void
SchedulerCommon::makeTaskGraphDoc(const DetailedTasks* /* dt*/,
                                        int            rank)
{
  if (!emit_taskgraph) {
    return;
  }

  if (!m_outPort->isOutputTimestep()){
    return;
  }
  // make sure to release this DOMDocument after finishing emitting the nodes
  m_graphDoc = ProblemSpec::createDocument("Uintah_TaskGraph");
  
  ProblemSpecP meta = m_graphDoc->appendChild("Meta");
  meta->appendElement("username", getenv("LOGNAME"));
  time_t t = time(NULL);
  meta->appendElement("date", ctime(&t));
  
  m_nodes = m_graphDoc->appendChild("Nodes");
  
  ProblemSpecP edgesElement = m_graphDoc->appendChild("Edges");
  
  for (unsigned i = 0; i < graphs.size(); i++) {
    DetailedTasks* dts = graphs[i]->getDetailedTasks();
    if (dts) {
      dts->emitEdges(edgesElement, rank);
    }
  }
}

//______________________________________________________________________
//
bool
SchedulerCommon::useInternalDeps()
{
  // keep track of internal dependencies only if it will emit
  // the taskgraphs (by default).
  return emit_taskgraph;
}

//______________________________________________________________________
//
void
SchedulerCommon::emitNode(const DetailedTask* task,
                                double        start,
                                double        duration,
                                double        execution_duration)
{  
  if (m_nodes == 0) {
    return;
  }

  ProblemSpecP node = m_nodes->appendChild("node");
  //m_nodes->appendChild(node);

  node->appendElement("name", task->getName());
  node->appendElement("start", start);
  node->appendElement("duration", duration);

  if (execution_duration > 0) {
    node->appendElement("execution_duration", execution_duration);
  }
}

//______________________________________________________________________
//
void
SchedulerCommon::finalizeNodes(int process /* = 0*/)
{
    if (m_graphDoc == 0){
      return;
    }

    if (m_outPort->isOutputTimestep()) {
      string timestep_dir(m_outPort->getLastTimestepOutputLocation());
      
      ostringstream fname;
      fname << "/taskgraph_" << setw(5) << setfill('0') << process << ".xml";
      string file_name(timestep_dir + fname.str());
      m_graphDoc->output(file_name.c_str());
    }
    
    //m_graphDoc->releaseDocument();
    //m_graphDoc = NULL;
    //m_nodes = NULL;
}

//______________________________________________________________________
//
void
SchedulerCommon::problemSetup(const ProblemSpecP&     prob_spec,
                                    SimulationStateP& state)
{
  d_sharedState = state;

  // Initializing trackingStartTime_ and trackingEndTime_ to default values
  // so that we do not crash when running MALLOC_STRICT.
  trackingStartTime_ = 1;
  trackingEndTime_ = 0;
  trackingVarsPrintLocation_ = PRINT_AFTER_EXEC;

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->getWithDefault("small_messages", d_useSmallMessages, true);
    
    if (d_useSmallMessages) {
      proc0cout << "   Using small, individual MPI messages (no message combining)\n";
    }
    else {
      proc0cout << "   Using large, combined MPI messages\n";
    }
    
    ProblemSpecP track = params->findBlock("VarTracker");
    if (track) {
      track->require("start_time", trackingStartTime_);
      track->require("end_time", trackingEndTime_);
      track->getWithDefault("level", trackingLevel_, -1);
      track->getWithDefault("start_index", trackingStartIndex_, IntVector(-9,-9,-9));
      track->getWithDefault("end_index", trackingEndIndex_, IntVector(-9,-9,-9));
      track->getWithDefault("patchid", trackingPatchID_, -1);

      if( d_myworld->myrank() == 0 ) {
        cout << "\n";
        cout << "-----------------------------------------------------------\n";
        cout << "-- Initializing VarTracker...\n";
        cout << "--  Running from time " << trackingStartTime_  << " to " << trackingEndTime_ << "\n";
        cout << "--  for indices: " << trackingStartIndex_ << " to " << trackingEndIndex_ << "\n";
      }

      ProblemSpecP location = track->findBlock("locations");
      if (location) {
        trackingVarsPrintLocation_ = 0;
        map<string, string> attributes;
        location->getAttributes(attributes);
        if (attributes["before_comm"] == "true") {
          trackingVarsPrintLocation_ |= PRINT_BEFORE_COMM;
          if (d_myworld->myrank() == 0) {
            cout << "--  Printing variable information before communication.\n";
          }
        }
        if (attributes["before_exec"] == "true") {
          trackingVarsPrintLocation_ |= PRINT_BEFORE_EXEC;
          if (d_myworld->myrank() == 0) {
            cout << "--  Printing variable information before task execution.\n";
          }
        }
        if (attributes["after_exec"] == "true") {
          trackingVarsPrintLocation_ |= PRINT_AFTER_EXEC;
          if (d_myworld->myrank() == 0) {
            cout << "--  Printing variable information after task execution.\n";
          }
        }
      }
      else {
        // "locations" not specified
        if (d_myworld->myrank() == 0) {
          cout << "--  Defaulting to printing variable information after task execution.\n";
        }
      }

      for (ProblemSpecP var=track->findBlock("var"); var != 0; var = var->findNextBlock("var")) {
        map<string,string> attributes;
        var->getAttributes(attributes);
        string name = attributes["label"];
        trackingVars_.push_back(name);
        string dw = attributes["dw"];

        if (dw == "OldDW") {
          trackingDWs_.push_back(Task::OldDW);
        }
        else if (dw == "NewDW") {
          trackingDWs_.push_back(Task::NewDW);
        }
        else if (dw == "CoarseNewDW") {
          trackingDWs_.push_back(Task::CoarseNewDW);
        }
        else if (dw == "CoarseOldDW") {
          trackingDWs_.push_back(Task::CoarseOldDW);
        }
        else if (dw == "ParentOldDW") {
          trackingDWs_.push_back(Task::ParentOldDW);
        }
        else if (dw == "ParentOldDW") {
          trackingDWs_.push_back(Task::ParentNewDW);
        }
        else {
          // This error message most likely can go away once the .ups validation is put into place:
          printf( "WARNING: Hit switch statement default... using NewDW... (This could possibly be"
                  "an error in input file specification.)\n" );
          trackingDWs_.push_back(Task::NewDW);
        }
        if( d_myworld->myrank() == 0 ) {
          cout << "--  Tracking variable '" << name << "' in DataWarehouse '" << dw << "'\n";
        }
      }

      for (ProblemSpecP task=track->findBlock("task"); task != 0; task = task->findNextBlock("task")) {
        map<string,string> attributes;
        task->getAttributes(attributes);
        string name = attributes["name"];
        trackingTasks_.push_back(name);
        if( d_myworld->myrank() == 0 ) { cout << "--  Tracking variables for specific task: " << name << "\n"; }
      }      
      if( d_myworld->myrank() == 0 ) {
        cout << "-----------------------------------------------------------\n\n";
      }
    }
    else { // Tracking not specified
      // This 'else' won't be necessary once the .ups files are validated... but for now.
      if( d_myworld->myrank() == 0 ) {
        cout << "<VarTracker> not specified in .ups file... no variable tracking will take place.\n";
      }
    }
  }
  noScrubVars_.insert("refineFlag");
  noScrubVars_.insert("refinePatchFlag");
}

//______________________________________________________________________
// handleError()
//
// The following routine is designed to only print out a given error
// once per error type per variable.  handleError is used by
// printTrackedVars() with each type of error ('errorPosition')
// condition specifically enumerated (by an integer running from 0 to 7).
//
// Returns true if the error message is displayed.
//
bool
handleError(       int     errorPosition,
             const string& errorMessage,
             const string& variableName )
{
  static vector< map<string,bool> * > errorsReported( 8 );

  map<string, bool> * varToReportedMap = errorsReported[ errorPosition ];

  if( varToReportedMap == NULL ) {
    varToReportedMap = new map<string, bool>;
    errorsReported[ errorPosition ] = varToReportedMap;
  }

  bool reported = (*varToReportedMap)[ variableName ];
  if( !reported ) {
    (*varToReportedMap)[ variableName ] = true;
    cout << errorMessage << "\n";
    return true;
  }
  return false;
}

//______________________________________________________________________
//
void
SchedulerCommon::printTrackedVars( DetailedTask* dt,
                                   int when )
{
  bool printedHeader = false;

  LoadBalancer* lb = getLoadBalancer();
 
  unsigned taskNum;
  for (taskNum = 0; taskNum < trackingTasks_.size(); taskNum++) {
    if (trackingTasks_[taskNum] == dt->getTask()->getName())
      break;
  }

  // Print for all tasks unless one is specified (but disclude DataArchiver tasks)
  if ((taskNum == trackingTasks_.size() && trackingTasks_.size() != 0) || 
      ((string(dt->getTask()->getName())).substr(0,12) == "DataArchiver")) {
    return;
  }

  if( d_sharedState && ( trackingStartTime_ > d_sharedState->getElapsedTime() ||
                         trackingEndTime_ < d_sharedState->getElapsedTime() ) ) {
    return;
  }

  for (int i = 0; i < (int) trackingVars_.size(); i++) {
    bool printedVarName = false;

    // that DW may not have been mapped....
    if (dt->getTask()->mapDataWarehouse(trackingDWs_[i]) < 0 || 
        dt->getTask()->mapDataWarehouse(trackingDWs_[i]) >= (int) dws.size()) {

      ostringstream mesg;
      mesg << "WARNING: VarTracker: Not printing requested variable (" << trackingVars_[i]
           << ") DW is out of range.\n";

      handleError( 0, mesg.str(), trackingVars_[i] );

      continue;
    }

    OnDemandDataWarehouseP dw = dws[dt->getTask()->mapDataWarehouse(trackingDWs_[i])];

    if (dw == 0) { // old on initialization timestep
      ostringstream mesg;

      mesg << "WARNING: VarTracker: Not printing requested variable (" << trackingVars_[i] 
           << ") because DW is NULL.  Requested DW was: " 
           << dt->getTask()->mapDataWarehouse(trackingDWs_[i]) << "\n";

      handleError( 1, mesg.str(), trackingVars_[i] );
      continue;
    }

    // get the level here, as the grid can be different between the old and new DW
    const Grid* grid = dw->getGrid();

    int levelnum;
    
    if (trackingLevel_ == -1) {
      levelnum = grid->numLevels() - 1;
    }
    else {
      levelnum = trackingLevel_;
      if (levelnum >= grid->numLevels()) {
        continue;
      }
    }

    const LevelP level = grid->getLevel(levelnum);
    const VarLabel* label = VarLabel::find(trackingVars_[i]);

    cout.precision(16);

    if (!label) {
      ostringstream mesg;
      mesg << "WARNING: VarTracker: Not printing requested variable (" << trackingVars_[i]
           << ") because label is NULL.\n";
      handleError( 2, mesg.str(), trackingVars_[i] );
      continue;
    }

    const PatchSubset* patches = dt->getPatches();
    
    // a once-per-proc task is liable to have multiple levels, and thus calls to getLevel(patches) will fail
    if ( dt->getTask()->getType() != Task::OncePerProc && (!patches || getLevel(patches)->getIndex() != levelnum) ) {
      ostringstream mesg;
      mesg << "WARNING: VarTracker: Not printing requested variable (" << trackingVars_[i]
           << ") because patch is non-standard.\n";
      handleError( 3, mesg.str(), trackingVars_[i] );
      continue;
    }
    
    //__________________________________
    //
    for (int p = 0; patches && p < patches->size(); p++) {

      const Patch* patch = patches->get(p);
      if (trackingPatchID_ != -1 && trackingPatchID_ != patch->getID()) {
        ostringstream mesg;
        mesg << "WARNING: VarTracker: Not printing requested variable (" << trackingVars_[i]
             << ") because patch ID does not match.\n" 
             << "            (Error only printed once.)\n"
             << "         Tracking Patch ID: " << trackingPatchID_ << ", patch id: " << patch->getID() << "\n";
        handleError( 4, mesg.str(), trackingVars_[i] );
        continue;
      }

      // don't print ghost patches (dw->get will yell at you)
      if ((trackingDWs_[i] == Task::OldDW && lb->getOldProcessorAssignment(0,patch,0) != d_myworld->myrank()) ||
          (trackingDWs_[i] == Task::NewDW && lb->getPatchwiseProcessorAssignment(patch) != d_myworld->myrank())) {
        continue;
      }

      const TypeDescription* td = label->typeDescription();
      Patch::VariableBasis basis = patch->translateTypeToBasis(td->getType(), false);
      
      IntVector start = Max(patch->getExtraLowIndex(basis, IntVector(0,0,0)), trackingStartIndex_);
      IntVector end   = Min(patch->getExtraHighIndex(basis, IntVector(0,0,0)), trackingEndIndex_);

      // loop over matls too
      for (int m = 0; m < d_sharedState->getNumMatls(); m++) {

        if (!dw->exists(label, m, patch)) {
          ostringstream mesg;
          mesg << "WARNING: VarTracker: Not printing requested variable (" << trackingVars_[i] 
               << ") because it does not exist in DW.\n"
               << "            Patch is: " << *patch << "\n";
          if( handleError( 5, mesg.str(), trackingVars_[i] ) ) {
            cout << "         DW contains (material: " << m << ")\n";
            dw->print();
          }
          continue;
        }
        if (!(start.x() < end.x() && start.y() < end.y() && start.z() < end.z())) {
          ostringstream mesg;
          mesg << "WARNING: VarTracker: Not printing requested variable (" << trackingVars_[i] 
               << ") because the start is greater than the end location:\n"
               << "start: " << start << "\n"
               << "end: " << start << "\n";
          handleError( 6, mesg.str(), trackingVars_[i] );
          continue;
        }
        if (td->getSubType()->getType() != TypeDescription::double_type &&
            td->getSubType()->getType() != TypeDescription::Vector) {

          // only allow *Variable<double> and *Variable<Vector> for now
          ostringstream mesg;
          mesg << "WARNING: VarTracker: Not printing requested variable (" << trackingVars_[i]
               << ") because its type is not supported:\n"
               << "             " << td->getName() << "\n";

          handleError( 7, mesg.str(), trackingVars_[i] );
          continue;
        }

        // pending the task that allocates the var, we may not have allocated it yet
        GridVariableBase* v;
        switch (td->getType()) {
          case TypeDescription::CCVariable :
          case TypeDescription::NCVariable :
          case TypeDescription::SFCXVariable :
          case TypeDescription::SFCYVariable :
          case TypeDescription::SFCZVariable :
            v = dynamic_cast<GridVariableBase*>(dw->d_varDB.get(label, m, patch));
            break;
          default :
            throw InternalError("Cannot track var type of non-grid-type", __FILE__, __LINE__);
            break;
        }
        
        start = Max(start, v->getLow());
        end   = Min(end,   v->getHigh());
        
        if (!(start.x() < end.x() && start.y() < end.y() && start.z() < end.z())) {
          continue;
        }

        if (!printedHeader) {
          string location;
          switch (when) {
          case PRINT_BEFORE_COMM: location = " before communication of "; break;
          case PRINT_BEFORE_EXEC: location = " before execution of "; break;
          case PRINT_AFTER_EXEC: location = " after execution of "; break;
          }
          cout << d_myworld->myrank() << location << *dt << endl;
          printedHeader = true;
        }
        
        if (!printedVarName) {
          cout << d_myworld->myrank() << "  Variable: " << trackingVars_[i] << ", DW " << dw->getID() << ", Patch " << patch->getID() << ", Matl " << m << endl;
          
          if (trackingVars_[i] == "rho_CC") {
            cout << "  RHO: " << dw->getID() << " original input " << trackingDWs_[i] << endl;
          }
        }
            
        switch (td->getSubType()->getType()) {
        case TypeDescription::double_type: 
        {
          GridVariable<double>* var = dynamic_cast<GridVariable<double>*>(v);
          
          for (int z = start.z(); z < end.z(); z++) {
            for (int y = start.y(); y < end.y(); y++) {
              cout << d_myworld->myrank() << "  ";
              for (int x = start.x(); x < end.x(); x++) {
                IntVector c(x,y,z);
                cout << " " << c << ": " << (*var)[c];
              }
              cout << endl;
            }
            cout << endl;
          }
        }
        break;
        case TypeDescription::Vector: 
        {
          GridVariable<Vector>* var = dynamic_cast<GridVariable<Vector>*>(v);
          
          for (int z = start.z(); z < end.z(); z++) {
            for (int y = start.y(); y < end.y(); y++) {
              cout << d_myworld->myrank() << "  ";
              for (int x = start.x(); x < end.x(); x++) {
                IntVector c(x,y,z);
                cout << " " << c << ": " << (*var)[c];
              }
              cout << endl;
            }
            cout << endl;
          }
        }
        break;
        default: break;
        } // end case variable type
      } // end for materials loop
    } // end for patches loop
  } // end for i : trackingVars.size()
} // end printTrackedVars()

//______________________________________________________________________
//
LoadBalancer*
SchedulerCommon::getLoadBalancer()
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   return lb;
}

//______________________________________________________________________
//
void
SchedulerCommon::addTaskGraph( Scheduler::tgType type )
{
  MALLOC_TRACE_TAG_SCOPE("SchedulerCommon::addTaskGraph");
  TaskGraph* tg = scinew TaskGraph(this, d_myworld, type);
  tg->initialize();
  graphs.push_back(tg);
}

//______________________________________________________________________
//
void
SchedulerCommon::addTask(       Task*        task,
                          const PatchSet*    patches,
			                    const MaterialSet* matls )
{
  MALLOC_TRACE_TAG_SCOPE("SchedulerCommon::addTask");

  // Save the DW map
  task->setMapping(dwmap);

  // if (d_myworld->myrank() == 1 || d_myworld->myrank() == d_myworld->size()-1)
  schedulercommon_dbg << d_myworld->myrank() << " adding Task: " << task->getName() << ", # patches: "
                      << (patches ? patches->size() : 0) << ", # matls: " << (matls ? matls->size() : 0) << endl;

  graphs[graphs.size()-1]->addTask(task, patches, matls);
  numTasks_++;

  // get the current level for the specified PatchSet to determine maxGhostCells and maxLevelOffset
  //   don't check these for output and restart tasks - patch and material sets are null then
  if (patches && matls) {
    int levelIndex = patches->getSubset(0)->get(0)->getLevel()->getIndex();

    // initialize or update max ghost cells for the current level
    std::map<int, int>::iterator mgc_iter;
    mgc_iter = maxGhostCells.find(levelIndex);
    int taskMGC = task->maxGhostCells;
    if (mgc_iter == maxGhostCells.end()) {
      maxGhostCells.insert(std::pair<int, int>(levelIndex, (taskMGC > 0 ? taskMGC : 0)));
    }
    else if (taskMGC > mgc_iter->second) {
      mgc_iter->second = task->maxGhostCells;
    }

    // initialize or update max level offset for the current level
    std::map<int, int>::iterator mlo_iter;
    mlo_iter = maxLevelOffsets.find(levelIndex);
    int taskMLO = task->maxLevelOffset;
    if (mlo_iter == maxLevelOffsets.end()) {
      maxLevelOffsets.insert(std::pair<int, int>(levelIndex, (taskMLO > 0 ? taskMLO : 0)));
    }
    else if (taskMLO > mlo_iter->second) {
      mlo_iter->second = task->maxLevelOffset;
    }
  }
  
  // add to init-requires.  These are the vars which require from the OldDW that we'll
  // need for checkpointing, switching, and the like.
  // In the case of treatAsOld Vars, we handle them because something external to the taskgraph
  // needs it that way (i.e., Regridding on a restart requires checkpointed refineFlags).
  for (const Task::Dependency* dep = task->getRequires(); dep != 0; dep = dep->next) {
    if (isOldDW(dep->mapDataWarehouse()) || treatAsOldVars_.find(dep->var->getName()) != treatAsOldVars_.end()) {
      d_initRequires.push_back(dep);
      d_initRequiredVars.insert(dep->var);
    }
  }

  // for the treat-as-old vars, go through the computes and add them.
  // we can (probably) safely assume that we'll avoid duplicates, since if they were inserted 
  // in the above, they wouldn't need to be marked as such
  for (const Task::Dependency* dep = task->getComputes(); dep != 0; dep = dep->next) {
    d_computedVars.insert(dep->var);

    if (treatAsOldVars_.find(dep->var->getName()) != treatAsOldVars_.end()) {
      d_initRequires.push_back(dep);
      d_initRequiredVars.insert(dep->var);
    }
  }

  //__________________________________
  // create reduction task if computes included one or more reduction vars
  for (const Task::Dependency* dep = task->getComputes(); dep != 0; dep = dep->next) {

    if (dep->var->typeDescription()->isReductionVariable()) {
      int levelidx = dep->reductionLevel ? dep->reductionLevel->getIndex() : -1;
      int dw = dep->mapDataWarehouse();

      if (dep->var->allowsMultipleComputes()) {
        if (schedulercommon_dbg.active()) {
          schedulercommon_dbg << d_myworld->myrank() << " Skipping Reduction task for multi compute variable: "
                              << dep->var->getName() << " on level " << levelidx << ", DW " << dw << '\n';
        }
        continue;
      }

      if (schedulercommon_dbg.active()) {
        schedulercommon_dbg << d_myworld->myrank() << " Creating Reduction task for variable: " << dep->var->getName()
                            << " on level " << levelidx << ", DW " << dw << '\n';
      }

      ostringstream taskname;
      taskname << "Reduction: " << dep->var->getName() << ", level " << levelidx << ", dw " << dw;

      Task* newtask = scinew Task(taskname.str(), Task::Reduction);

      int dwmap[Task::TotalDWs];

      for (int i = 0; i < Task::TotalDWs; i++) {
        dwmap[i] = Task::InvalidDW;
      }

      dwmap[Task::OldDW] = Task::NoDW;
      dwmap[Task::NewDW] = dw;
      newtask->setMapping(dwmap);

      if (dep->matls != 0) {
        newtask->modifies(dep->var, dep->reductionLevel, dep->matls, Task::OutOfDomain);
        for (int i = 0; i < dep->matls->size(); i++) {
          int maltIdx = dep->matls->get(i);
          VarLabelMatl<Level> key(dep->var, maltIdx, dep->reductionLevel);
          reductionTasks[key] = newtask;
        }
      }
      else {
        for (int m = 0; m < task->getMaterialSet()->size(); m++) {
          newtask->modifies(dep->var, dep->reductionLevel, task->getMaterialSet()->getSubset(m), Task::OutOfDomain);
          for (int i = 0; i < task->getMaterialSet()->getSubset(m)->size(); i++) {
            int maltIdx = task->getMaterialSet()->getSubset(m)->get(i);
            VarLabelMatl<Level> key(dep->var, maltIdx, dep->reductionLevel);
            reductionTasks[key] = newtask;
          }
        }
      }

      graphs[graphs.size() - 1]->addTask(newtask, 0, 0);
      numTasks_++;
    }
  }
}

//______________________________________________________________________
//
void
SchedulerCommon::releaseLoadBalancer()
{
  releasePort("load balancer");
}

//______________________________________________________________________
//
void
SchedulerCommon::initialize( int numOldDW /* = 1 */,
                             int numNewDW /* = 1 */ )
{

  // doesn't really do anything except initialize/clear the taskgraph
  //   if the default parameter values are used
  int numDW = numOldDW + numNewDW;
  int oldnum = (int)dws.size();

  // in AMR cases we will often need to move from many new DWs to one.  In those cases, move the last NewDW to be the next new one.
  if (oldnum - numOldDWs > 1) {
    dws[numDW - 1] = dws[oldnum - 1];
  }

  // Clear out the data warehouse so that memory will be freed
  for (int i = numDW; i < oldnum; i++) {
    dws[i] = 0;
  }

  dws.resize(numDW);
  for (; oldnum < numDW; oldnum++) {
    dws[oldnum] = 0;
  }

  numOldDWs = numOldDW;

  // clear the taskgraphs, and set the first one
  for (unsigned i = 0; i < graphs.size(); i++) {
    delete graphs[i];
  }

  numParticleGhostCells_ = 0;

  graphs.clear();

  d_initRequires.clear();
  d_initRequiredVars.clear();
  d_computedVars.clear();
  numTasks_ = 0;

  maxGhostCells.clear();
  maxLevelOffsets.clear();

  reductionTasks.clear();
  addTaskGraph(NormalTaskGraph);

}

//______________________________________________________________________
//
void
SchedulerCommon::setParentDWs( DataWarehouse* parent_old_dw,
                               DataWarehouse* parent_new_dw )
{
  OnDemandDataWarehouse* pold = dynamic_cast<OnDemandDataWarehouse*>(parent_old_dw);
  OnDemandDataWarehouse* pnew = dynamic_cast<OnDemandDataWarehouse*>(parent_new_dw);
  if (parent_old_dw && parent_new_dw) {
    ASSERT(pold != 0);
    ASSERT(pnew != 0);
    ASSERT(numOldDWs > 2);
    dws[0] = pold;
    dws[1] = pnew;
  }
}

//______________________________________________________________________
//
void
SchedulerCommon::clearMappings()
{
  for(int i=0;i<Task::TotalDWs;i++) {
    dwmap[i]=-1;
  }
}

//______________________________________________________________________
//
void
SchedulerCommon::mapDataWarehouse( Task::WhichDW which,
                                   int dwTag )
{
  ASSERTRANGE(which, 0, Task::TotalDWs);
  ASSERTRANGE(dwTag, 0, static_cast<int>(dws.size()));
  dwmap[which]=dwTag;
}

//______________________________________________________________________
//
DataWarehouse*
SchedulerCommon::get_dw( int idx )
{
  ASSERTRANGE(idx, 0, static_cast<int>(dws.size()));
  return dws[idx].get_rep();
}

//______________________________________________________________________
//
DataWarehouse*
SchedulerCommon::getLastDW( void )
{
  return get_dw(static_cast<int>(dws.size()) - 1);
}

//______________________________________________________________________
//
void
SchedulerCommon::advanceDataWarehouse( const GridP& grid,
                                             bool initialization /*=false*/ )
{
  schedulercommon_dbg << "advanceDataWarehouse, numDWs = " << dws.size() << '\n';
  ASSERT(dws.size() >= 2);
  // The last becomes last old, and the rest are new
  dws[numOldDWs - 1] = dws[dws.size() - 1];

  if (dws.size() == 2 && dws[0] == 0) {
    // first datawarehouse -- indicate that it is the "initialization" dw.
    int generation = d_generation++;
    dws[1] = scinew OnDemandDataWarehouse(d_myworld, this, generation, grid, true /* initialization dw */);
  }
  else {
    for (int i = numOldDWs; i < static_cast<int>(dws.size()); i++) {
      // in AMR initial cases, you can still be in initialization when you advance again
      replaceDataWarehouse(i, grid, initialization);
    }
  }
}

//______________________________________________________________________
//
void 
SchedulerCommon::fillDataWarehouses( const GridP& grid )
{
  MALLOC_TRACE_TAG_SCOPE("SchedulerCommon::fillDatawarehouses");
  for (int i = numOldDWs; i < static_cast<int>(dws.size()); i++) {
    if (!dws[i]) {
      replaceDataWarehouse(i, grid);
    }
  }
}

//______________________________________________________________________
//
void
SchedulerCommon::replaceDataWarehouse(       int index,
                                       const GridP& grid,
                                             bool initialization /*=false*/)
{
  dws[index] = scinew OnDemandDataWarehouse(d_myworld, this, d_generation++, grid, initialization );
  if (initialization) {
    return;
  }
  for (unsigned i = 0; i < graphs.size(); i++) {
    DetailedTasks* dts = graphs[i]->getDetailedTasks();
    if (dts) {
      dts->copyoutDWKeyDatabase(dws[index]);
    }
  }
  dws[index]->doReserve();
}

//______________________________________________________________________
//
void 
SchedulerCommon::setRestartable( bool restartable )
{
  this->restartable = restartable;
}

//______________________________________________________________________
//
const vector<const Patch*>*
SchedulerCommon::getSuperPatchExtents( const VarLabel*        label,
                                             int              matlIndex,
                                       const Patch*           patch,
                                             Ghost::GhostType requestedGType,
                                             int              requestedNumGCells,
                                             IntVector&       requiredLow,
                                             IntVector&       requiredHigh,
                                             IntVector&       requestedLow,
                                             IntVector&       requestedHigh ) const
{
  const SuperPatch* connectedPatchGroup = m_locallyComputedPatchVarMap->getConnectedPatchGroup(patch);

  if (connectedPatchGroup == 0) {
    return 0;
  }

  SuperPatch::Region requestedExtents = connectedPatchGroup->getRegion();
  SuperPatch::Region requiredExtents = connectedPatchGroup->getRegion();

  // expand to cover the entire connected patch group
  bool containsGivenPatch = false;
  for (unsigned int i = 0; i < connectedPatchGroup->getBoxes().size(); i++) {
    // get the minimum extents containing both the expected ghost cells
    // to be needed and the given ghost cells.
    const Patch* memberPatch = connectedPatchGroup->getBoxes()[i];

    Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), true);

    IntVector lowOffset = IntVector(0, 0, 0), highOffset = IntVector(0, 0, 0);

    //set requiredLow and requiredHigh as extents without ghost cells
    memberPatch->computeExtents(basis, label->getBoundaryLayer(), lowOffset, highOffset, requiredLow, requiredHigh);

    //compute ghost cell offsets
    Patch::getGhostOffsets(basis, requestedGType, requestedNumGCells, lowOffset, highOffset);

    //set requestedLow and requestedHigh as extents with ghost cells
    memberPatch->computeExtents(basis, label->getBoundaryLayer(), lowOffset, highOffset, requestedLow, requestedHigh);

    SuperPatch::Region requiredRegion = SuperPatch::Region(requiredLow, requiredHigh);
    requiredExtents = requiredExtents.enclosingRegion(requiredRegion);
    SuperPatch::Region requestedRegion = SuperPatch::Region(requestedLow, requestedHigh);
    requestedExtents = requestedExtents.enclosingRegion(requestedRegion);

    if (memberPatch == patch) {
      containsGivenPatch = true;
    }
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

//______________________________________________________________________
//
void
SchedulerCommon::logMemoryUse()
{
  if (!memlogfile) {
    ostringstream fname;
    fname << "uintah_memuse.log.p" << setw(5) << setfill('0') << d_myworld->myrank() << "." << d_myworld->size();
    memlogfile = scinew ofstream(fname.str().c_str());
    if (!*memlogfile) {
      cerr << "Error opening file: " << fname.str() << '\n';
    }
  }

  *memlogfile << '\n';
  unsigned long total = 0;

  for (int i = 0; i < (int)dws.size(); i++) {
    char* name;
    if (i == 0) {
      name = const_cast<char*>("OldDW");
    }
    else if (i == (int)dws.size() - 1) {
      name = const_cast<char*>("NewDW");
    }
    else {
      name = const_cast<char*>("IntermediateDW");
    }

    if (dws[i]) {
      dws[i]->logMemoryUse(*memlogfile, total, name);
    }

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

//______________________________________________________________________
//
// Makes and returns a map that maps strings to VarLabels of
// that name and a list of material indices for which that
// variable is valid (according to d_allcomps in graph).
Scheduler::VarLabelMaterialMap*
SchedulerCommon::makeVarLabelMaterialMap()
{
  VarLabelMaterialMap* result = scinew VarLabelMaterialMap;
  for (unsigned i = 0; i < graphs.size(); i++) {
    graphs[i]->makeVarLabelMaterialMap(result);
  }
  return result;
}

//______________________________________________________________________
//     
void
SchedulerCommon::doEmitTaskGraphDocs()
{
  emit_taskgraph=true;
}

//______________________________________________________________________
//
void
SchedulerCommon::compile()
{
  TAU_PROFILE("SchedulerCommon::compile()", " ", TAU_USER); 

  GridP grid = const_cast<Grid*>(getLastDW()->getGrid());
  GridP oldGrid;
  
  if (dws[0]) {
    oldGrid = const_cast<Grid*>(get_dw(0)->getGrid());
  }
  
  if(numTasks_ > 0){

    schedulercommon_dbg << d_myworld->myrank() << " SchedulerCommon starting compile\n";
    
    // pass the first to the rest, so we can share the scrubcountTable
    DetailedTasks* first = 0;
    for (unsigned i = 0; i < graphs.size(); i++) {
      if (graphs.size() > 1) {
        schedulercommon_dbg << d_myworld->myrank() << "  Compiling graph#" << i << " of " << graphs.size() << endl;
      }
      
      DetailedTasks* dts = graphs[i]->createDetailedTasks(useInternalDeps(), first, grid, oldGrid);
      
      if (!first) {
        first = dts;
      }
    }
    verifyChecksum();
    schedulercommon_dbg << d_myworld->myrank() << " SchedulerCommon finished compile\n";
  }

  m_locallyComputedPatchVarMap->reset();

#if 1
  for (int i = 0; i < grid->numLevels(); i++) {
    const PatchSubset* patches = getLoadBalancer()->getPerProcessorPatchSet(grid->getLevel(i))->getSubset(d_myworld->myrank());
    if (patches->size() > 0) {
      m_locallyComputedPatchVarMap->addComputedPatchSet(patches);
    }
  }
#else
  for (unsigned i = 0; i < graphs.size(); i++) { 
    DetailedTasks* dts = graphs[i]->getDetailedTasks();
    
    if (dts != 0) {    
      // figure out the locally computed patches for each variable.
      for (int i = 0; i < dts->numLocalTasks(); i++) {
        const DetailedTask* dt = dts->localTask(i);
        
        for(const Task::Dependency* comp = dt->getTask()->getComputes();comp != 0; comp = comp->next){
        
          if (comp->var->typeDescription()->getType() != TypeDescription::ReductionVariable) {
            constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(dt->getPatches());
            m_locallyComputedPatchVarMap->addComputedPatchSet(patches.get_rep());
          }
        }
      }
    }
  }
#endif
  for(unsigned int dw=0;dw<dws.size();dw++) {
    if (dws[dw].get_rep()) {
      for (unsigned i = 0; i < graphs.size(); i++) { 
        DetailedTasks* dts = graphs[i]->getDetailedTasks();
        dts->copyoutDWKeyDatabase(dws[dw]);
      }
      dws[dw]->doReserve();
    }
  }
  m_locallyComputedPatchVarMap->makeGroups();
}

//______________________________________________________________________
//
bool
SchedulerCommon::isOldDW( int idx ) const
{
  ASSERTRANGE(idx, 0, static_cast<int>(dws.size()));
  return idx < numOldDWs;
}

//______________________________________________________________________
//
bool
SchedulerCommon::isNewDW( int idx ) const
{
  ASSERTRANGE(idx, 0, static_cast<int>(dws.size()));
  return idx >= numOldDWs;
}

//______________________________________________________________________
//
void
SchedulerCommon::finalizeTimestep()
{
  finalizeNodes(d_myworld->myrank());
  for (unsigned int i = numOldDWs; i < dws.size(); i++) {
    dws[i]->finalize();
  }
}

//______________________________________________________________________
//
void
SchedulerCommon::scheduleAndDoDataCopy( const GridP&               grid,
                                              SimulationInterface* sim )
{
  
  TAU_PROFILE("SchedulerCommon::scheduleAndDoDataCopy()", " ", TAU_USER);
  TAU_PROFILE_TIMER(sched_timer,"schedule", "", TAU_USER);
  TAU_PROFILE_START(sched_timer);

  double start = Time::currentSeconds();
  // TODO - use the current initReqs and push them back, instead of doing this...
  // clear the old list of vars and matls
  for (unsigned i = 0; i < label_matls_.size(); i++) {
    for (label_matl_map::iterator iter = label_matls_[i].begin(); iter != label_matls_[i].end(); iter++) {
      if (iter->second->removeReference()) {
        delete iter->second;
      }
    }
  }

  label_matls_.clear();
  label_matls_.resize(grid->numLevels());

  // produce a map from all tasks' requires from the Old DW.  Store the varlabel and matls
  // TODO - only do this ONCE.
  for (unsigned t = 0; t < graphs.size(); t++) {
    TaskGraph* tg = graphs[t];
    for (int i = 0; i < tg->getNumTasks(); i++) {
      Task* task = tg->getTask(i);
      if (task->getType() == Task::Output) {
        continue;
      }

      for (const Task::Dependency* dep = task->getRequires(); dep != 0; dep = dep->next) {
        bool copyThisVar = dep->whichdw == Task::OldDW;
        // override to manually copy a var
        if (!copyThisVar) {
          if (copyDataVars_.find(dep->var->getName()) != copyDataVars_.end()) {
            copyThisVar = true;
          }
        }

        // Overide the logic above.  There are PerPatch variables that cannot/shouldn't be copied to the new grid,
        // for example PerPatch<FileInfo>.
        if (notCopyDataVars_.count(dep->var->getName()) > 0) {
          copyThisVar = false;
        }

        if (copyThisVar) {
          if (dep->var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
            // we will take care of reduction variables in a different section
            continue;
          }

          // check the level on the case where variables are only computed on certain levels
          const PatchSet* ps = task->getPatchSet();
          int level = -1;
          if (dep->patches) {        // just in case the task is over multiple levels...
            level = getLevel(dep->patches)->getIndex();
          }
          else if (ps) {
            level = getLevel(ps)->getIndex();
          }

          // we don't want data with an invalid level, or requiring from a different level (remember, we are
          // using an old task graph).  That willbe copied later (and chances are, it's to modify anyway).
          if (level == -1 || level > grid->numLevels() - 1 || dep->patches_dom == Task::CoarseLevel
              || dep->patches_dom == Task::FineLevel) {
            continue;
          }

          const MaterialSubset* matSubset = (dep->matls != 0) ? dep->matls : dep->task->getMaterialSet()->getUnion();

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
  this->advanceDataWarehouse(grid, true);
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
  vector<Handle<PatchSet> > refinePatchSets(grid->numLevels(), (PatchSet*)0);
  vector<Handle<PatchSet> > copyPatchSets(grid->numLevels(), (PatchSet*)0);
  SchedulerP sched(dynamic_cast<Scheduler*>(this));

  d_sharedState->setCopyDataTimestep(true);

  for (int L = 0; L < grid->numLevels(); L++) {
    LevelP newLevel = grid->getLevel(L);
    //const PatchSubset  *patches = getLoadBalancer()->getPerProcessorPatchSet(newLevel)->getSubset(d_myworld->myrank());
    if (L > 0) {

      if (L >= oldGrid->numLevels()) {
        // new level - refine everywhere
        refinePatchSets[L] = const_cast<PatchSet*>(newLevel->eachPatch());
        copyPatchSets[L] = scinew PatchSet;
      }

      // find patches with new space - but temporarily, refine everywhere... 
      else if (L < oldGrid->numLevels()) {
        refinePatchSets[L] = scinew PatchSet;
        copyPatchSets[L] = scinew PatchSet;

        vector<int> myPatchIDs;
        LevelP oldLevel = oldDataWarehouse->getGrid()->getLevel(L);

        // go through the patches, and find if there are patches that weren't entirely 
        // covered by patches on the old grid, and interpolate them.  
        // then after, copy the data, and if necessary, overwrite interpolated data
        const PatchSubset *ps = getLoadBalancer()->getPerProcessorPatchSet(newLevel)->getSubset(d_myworld->myrank());

        // for each patch I own
        for (int p = 0; p < ps->size(); p++) {
          const Patch *newPatch = ps->get(p);

          // get the low/high for what we'll need to get
          IntVector lowIndex, highIndex;
          //newPatch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0), Ghost::None, 0, lowIndex, highIndex);
          lowIndex = newPatch->getCellLowIndex();
          highIndex = newPatch->getCellHighIndex();

          // find if area on the new patch was not covered by the old patches
          IntVector dist = highIndex - lowIndex;
          int totalCells = dist.x() * dist.y() * dist.z();
          int sum = 0;
          Patch::selectType oldPatches;
          oldLevel->selectPatches(lowIndex, highIndex, oldPatches);

          //compute volume of overlapping regions
          for (int old = 0; old < oldPatches.size(); old++) {

            const Patch* oldPatch = oldPatches[old];
            IntVector oldLow = oldPatch->getCellLowIndex();
            IntVector oldHigh = oldPatch->getCellHighIndex();

            IntVector low = Max(oldLow, lowIndex);
            IntVector high = Min(oldHigh, highIndex);
            IntVector dist = high - low;
            sum += dist.x() * dist.y() * dist.z();
          }  // for oldPatches  

          if (sum != totalCells) {
            if (Uintah::Parallel::usingMPI()) {
              myPatchIDs.push_back(newPatch->getID());
            }
            else {
              refinePatchSets[L]->add(newPatch);
            }
          }
          else {
            if (!Uintah::Parallel::usingMPI()) {
              copyPatchSets[L]->add(newPatch);
            }
          }
        }  // for patch

        if (Uintah::Parallel::usingMPI()) {
          //Gather size from all processors
          int mycount = myPatchIDs.size();
          vector<int> counts(d_myworld->size());
          MPI_Allgather(&mycount, 1, MPI_INT, &counts[0], 1, MPI_INT, d_myworld->getComm());

          //compute recieve array offset and size
          vector<int> displs(d_myworld->size());
          int pos = 0;

          for (int p = 0; p < d_myworld->size(); p++) {
            displs[p] = pos;
            pos += counts[p];
          }

          vector<int> allPatchIDs(pos);  //receive array;
          MPI_Allgatherv(&myPatchIDs[0], counts[d_myworld->myrank()], MPI_INT, &allPatchIDs[0], &counts[0], &displs[0], MPI_INT,
                         d_myworld->getComm());
          //make refinePatchSets from patch ids
          set<int> allPatchIDset(allPatchIDs.begin(), allPatchIDs.end());

          for (Level::patchIterator iter = newLevel->patchesBegin(); iter != newLevel->patchesEnd(); ++iter) {
            Patch* newPatch = *iter;
            if (allPatchIDset.find(newPatch->getID()) != allPatchIDset.end()) {
              refinePatchSets[L]->add(newPatch);
            }
            else {
              copyPatchSets[L]->add(newPatch);
            }
          }
        }  // using MPI
      }

      if (refinePatchSets[L]->size() > 0) {
        schedulercommon_dbg << d_myworld->myrank() << "  Calling scheduleRefine for patches " << *refinePatchSets[L].get_rep()
                            << endl;
        sim->scheduleRefine(refinePatchSets[L].get_rep(), sched);
      }

    }
    else {
      refinePatchSets[L] = scinew PatchSet;
      copyPatchSets[L] = const_cast<PatchSet*>(newLevel->eachPatch());
    }

    //__________________________________
    //  Scheduling for copyDataToNewGrid
    if (copyPatchSets[L]->size() > 0) {
      dataTasks.push_back(scinew Task("SchedulerCommon::copyDataToNewGrid", this, &SchedulerCommon::copyDataToNewGrid));

      for (label_matl_map::iterator iter = label_matls_[L].begin(); iter != label_matls_[L].end(); iter++) {
        const VarLabel* var = iter->first;
        MaterialSubset* matls = iter->second;

        dataTasks.back()->requires(Task::OldDW, var, 0, Task::OtherGridDomain, matls, Task::NormalDomain, Ghost::None, 0);
        schedulercommon_dbg << "  Scheduling copy for var " << *var << " matl " << *matls << " Copies: "
                            << *copyPatchSets[L].get_rep() << endl;
        dataTasks.back()->computes(var, matls);
      }
      addTask(dataTasks.back(), copyPatchSets[L].get_rep(), d_sharedState->allMaterials());
    }

    //__________________________________
    //  Scheduling for modifyDataOnNewGrid
    if (refinePatchSets[L]->size() > 0) {
      dataTasks.push_back(scinew Task("SchedulerCommon::modifyDataOnNewGrid", this, &SchedulerCommon::copyDataToNewGrid));

      for (label_matl_map::iterator iter = label_matls_[L].begin(); iter != label_matls_[L].end(); iter++) {
        const VarLabel* var = iter->first;
        MaterialSubset* matls = iter->second;

        dataTasks.back()->requires(Task::OldDW, var, 0, Task::OtherGridDomain, matls, Task::NormalDomain, Ghost::None, 0);
        schedulercommon_dbg << "  Scheduling modify for var " << *var << " matl " << *matls << " Modifies: "
                            << *refinePatchSets[L].get_rep() << endl;
        dataTasks.back()->modifies(var, matls);
      }
      addTask(dataTasks.back(), refinePatchSets[L].get_rep(), d_sharedState->allMaterials());
    }

    //__________________________________
    //  Component's shedule for refineInterfae
    if (L > 0) {
      sim->scheduleRefineInterface(newLevel, sched, 0, 1);
    }
  }

  // set so the load balancer will make an adequate neighborhood, as the default
  // neighborhood isn't good enough for the copy data timestep
  d_sharedState->setCopyDataTimestep(true);  //-- do we still need this?  - BJW

#if !defined( DISABLE_SCI_MALLOC )
  const char* tag = AllocatorSetDefaultTag("DoDataCopy");
#endif
  this->compile();

  d_sharedState->regriddingCompilationTime += Time::currentSeconds() - start;

  TAU_PROFILE_STOP(sched_timer);
  TAU_PROFILE_TIMER(copy_timer,"copy", "", TAU_USER);
  TAU_PROFILE_START(copy_timer);

  // save these and restore them, since the next execute will append the scheduler's, and we don't want to.
  double executeTime = d_sharedState->taskExecTime;
  double globalCommTime = d_sharedState->taskGlobalCommTime;
  double localCommTime = d_sharedState->taskLocalCommTime;

  start = Time::currentSeconds();
  this->execute();

#if !defined( DISABLE_SCI_MALLOC )
  AllocatorSetDefaultTag(tag);
#endif

  //__________________________________
  // copy reduction variables to the new_dw
  vector<VarLabelMatl<Level> > levelVariableInfo;
  oldDataWarehouse->getVarLabelMatlLevelTriples(levelVariableInfo);

  newDataWarehouse->unfinalize();
  for (unsigned int i = 0; i < levelVariableInfo.size(); i++) {
    VarLabelMatl<Level> currentReductionVar = levelVariableInfo[i];

    if (currentReductionVar.label_->typeDescription()->isReductionVariable()) {

      // cout << "REDUNCTION:  Label(" << setw(15) << currentReductionVar.label_->getName() << "): Patch(" << reinterpret_cast<int>(currentReductionVar.level_) << "): Material(" << currentReductionVar.matlIndex_ << ")" << endl; 
      const Level* oldLevel = currentReductionVar.domain_;
      const Level* newLevel = NULL;
      if (oldLevel && oldLevel->getIndex() < grid->numLevels()) {

        if (oldLevel->getIndex() >= grid->numLevels()) {
          // the new grid no longer has this level
          continue;
        }
        newLevel = (newDataWarehouse->getGrid()->getLevel(oldLevel->getIndex())).get_rep();
      }

      //  Either both levels need to be null or both need to exist (null levels mean global data)
      if (!oldLevel || newLevel) {
        ReductionVariableBase* v =
            dynamic_cast<ReductionVariableBase*>(currentReductionVar.label_->typeDescription()->createInstance());

        oldDataWarehouse->get(*v, currentReductionVar.label_, currentReductionVar.domain_, currentReductionVar.matlIndex_);
        ;
        newDataWarehouse->put(*v, currentReductionVar.label_, newLevel, currentReductionVar.matlIndex_);
        delete v;  // copied on the put command
      }
    }
  }

  newDataWarehouse->refinalize();

  d_sharedState->regriddingCopyDataTime += Time::currentSeconds() - start;
  d_sharedState->taskExecTime = executeTime;
  d_sharedState->taskGlobalCommTime = globalCommTime;
  d_sharedState->taskLocalCommTime = localCommTime;

  TAU_PROFILE_STOP(copy_timer);

  d_sharedState->setCopyDataTimestep(false);
}

//______________________________________________________________________
//
void
SchedulerCommon::copyDataToNewGrid( const ProcessorGroup*,
                                    const PatchSubset*     patches,
                                    const MaterialSubset*  matls,
                                          DataWarehouse*   old_dw,
                                          DataWarehouse*   new_dw )
{
  schedulercommon_dbg << "SchedulerCommon::copyDataToNewGrid() BGN on patches " << *patches << endl;

  OnDemandDataWarehouse* oldDataWarehouse = dynamic_cast<OnDemandDataWarehouse*>(old_dw);
  OnDemandDataWarehouse* newDataWarehouse = dynamic_cast<OnDemandDataWarehouse*>(new_dw);

  // For each patch in the patch subset which contains patches in the new grid
  for (int p = 0; p < patches->size(); p++) {
    const Patch* newPatch = patches->get(p);
    const Level* newLevel = newPatch->getLevel();

    // to create once per matl instead of once per matl-var
    vector<ParticleSubset*> oldsubsets(d_sharedState->getNumMatls()), newsubsets(d_sharedState->getNumMatls());

    // If there is a level that didn't exist, we don't need to copy it
    if (newLevel->getIndex() >= oldDataWarehouse->getGrid()->numLevels()) {
      continue;
    }

    // find old patches associated with this patch
    LevelP oldLevel = oldDataWarehouse->getGrid()->getLevel(newLevel->getIndex());

    //__________________________________
    //  Loop over Var labels
    for (label_matl_map::iterator iter = label_matls_[oldLevel->getIndex()].begin();
        iter != label_matls_[oldLevel->getIndex()].end(); iter++) {
      const VarLabel* label = iter->first;
      MaterialSubset* var_matls = iter->second;

      // get the low/high for what we'll need to get
      Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), true);
      IntVector newLowIndex, newHighIndex;
      newPatch->computeVariableExtents(basis, IntVector(0, 0, 0), Ghost::None, 0, newLowIndex, newHighIndex);

      //__________________________________
      //  Loop over materials
      for (int m = 0; m < var_matls->size(); m++) {
        int matl = var_matls->get(m);

        if (!matls->contains(matl)) {
          //cout << "We are skipping material " << currentVar.matlIndex_ << endl;
          continue;
        }

        //__________________________________
        //  Grid Variables
        if (label->typeDescription()->getType() != TypeDescription::ParticleVariable) {
          Patch::selectType oldPatches;
          oldLevel->selectPatches(newLowIndex, newHighIndex, oldPatches);

          for (int oldIdx = 0; oldIdx < oldPatches.size(); oldIdx++) {
            const Patch* oldPatch = oldPatches[oldIdx];

            if (!oldDataWarehouse->exists(label, matl, oldPatch)) {
              continue;  // see comment about oldPatchToTest in ScheduleAndDoDataCopy
            }

            IntVector oldLowIndex;
            IntVector oldHighIndex;

            if (newLevel->getIndex() > 0) {
              oldLowIndex = oldPatch->getLowIndexWithDomainLayer(basis);
              oldHighIndex = oldPatch->getHighIndexWithDomainLayer(basis);
            }
            else {
              oldLowIndex = oldPatch->getExtraLowIndex(basis, label->getBoundaryLayer());
              oldHighIndex = oldPatch->getExtraHighIndex(basis, label->getBoundaryLayer());
            }

            IntVector copyLowIndex = Max(newLowIndex, oldLowIndex);
            IntVector copyHighIndex = Min(newHighIndex, oldHighIndex);

            // based on the selectPatches above, we might have patches we don't want to use, so prune them here.
            if (copyLowIndex.x() >= copyHighIndex.x() || copyLowIndex.y() >= copyHighIndex.y()
                || copyLowIndex.z() >= copyHighIndex.z()) {
              continue;
            }

            switch (label->typeDescription()->getType()) {
              case TypeDescription::NCVariable :
              case TypeDescription::CCVariable :
              case TypeDescription::SFCXVariable :
              case TypeDescription::SFCYVariable :
              case TypeDescription::SFCZVariable : {
                // bulletproofing
                if (!oldDataWarehouse->exists(label, matl, oldPatch)) {
                  SCI_THROW(UnknownVariable(label->getName(), oldDataWarehouse->getID(), oldPatch, matl, "in copyDataTo GridVariableBase", __FILE__, __LINE__));
                }

                vector<Variable *> varlist;
                oldDataWarehouse->d_varDB.getlist(label, matl, oldPatch, varlist);
                GridVariableBase* v = NULL;

                IntVector srclow = copyLowIndex;
                IntVector srchigh = copyHighIndex;

                for (unsigned int i = 0; i < varlist.size(); ++i) {
                  v = dynamic_cast<GridVariableBase*>(varlist[i]);

                  ASSERT(v->getBasePointer() != 0);

                  //restrict copy to data range
                  srclow = Max(copyLowIndex, v->getLow());
                  srchigh = Min(copyHighIndex, v->getHigh());

                  if (srclow.x() >= srchigh.x() || srclow.y() >= srchigh.y() || srclow.z() >= srchigh.z()) {
                    continue;
                  }

                  if (!newDataWarehouse->exists(label, matl, newPatch)) {

                    GridVariableBase* newVariable = v->cloneType();
                    newVariable->rewindow(newLowIndex, newHighIndex);
                    newVariable->copyPatch(v, srclow, srchigh);
                    newDataWarehouse->d_varDB.put(label, matl, newPatch, newVariable, isCopyDataTimestep(), false);

                  }
                  else {
                    GridVariableBase* newVariable = dynamic_cast<GridVariableBase*>(newDataWarehouse->d_varDB.get(label, matl,
                                                                                                                  newPatch));
                    // make sure it exists in the right region (it might be ghost data)
                    newVariable->rewindow(newLowIndex, newHighIndex);

                    if (oldPatch->isVirtual()) {
                      // it can happen where the old patch was virtual and this is not
                      GridVariableBase* tmpVar = newVariable->cloneType();
                      tmpVar->copyPointer(*v);
                      tmpVar->offset(oldPatch->getVirtualOffset());
                      newVariable->copyPatch(tmpVar, srclow, srchigh);
                      delete tmpVar;
                    }
                    else {
                      newVariable->copyPatch(v, srclow, srchigh);

                    }
                  }
                }
              }
                break;
              case TypeDescription::PerPatch : {
              }
                break;
              default :
                SCI_THROW(InternalError("Unknown variable type in copyData: "+label->getName(), __FILE__, __LINE__));
            }  // end switch
          }  // end oldPatches
        }
        else {
          //__________________________________
          //  Particle Variables
          ParticleSubset* oldsub = oldsubsets[matl];
          if (!oldsub) {
            // collect the particles from the range encompassing this patch.  Use interior cells since
            // extracells aren't collected across processors in the data copy, and they don't matter
            // for particles anyhow (but we will have to reset the bounds to copy the data)
            oldsub = oldDataWarehouse->getParticleSubset(matl, newPatch->getLowIndexWithDomainLayer(Patch::CellBased),
                                                         newPatch->getHighIndexWithDomainLayer(Patch::CellBased), newPatch,
                                                         reloc_new_posLabel_, oldLevel.get_rep());
            oldsubsets[matl] = oldsub;
            oldsub->addReference();
          }

          ParticleSubset* newsub = newsubsets[matl];
          // it might have been created in Refine
          if (!newsub) {
            if (!newDataWarehouse->haveParticleSubset(matl, newPatch)) {
              newsub = newDataWarehouse->createParticleSubset(oldsub->numParticles(), matl, newPatch);
            }
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
            ParticleSubset* tempset = scinew ParticleSubset(oldsub->numParticles(), matl, newPatch,
                                                            newPatch->getExtraCellLowIndex(), newPatch->getExtraCellHighIndex());
            const_cast<ParticleVariableBase*>(&var->getBaseRep())->setParticleSubset(tempset);
            newv->copyData(&var->getBaseRep());
            delete var;  //pset and tempset are deleted with it.
          }
          newDataWarehouse->put(*newv, label, true);
          delete newv;  // the container is copied
        }
      }  // end matls
    }  // end label_matls

    for (unsigned i = 0; i < oldsubsets.size(); i++) {
      if (oldsubsets[i] && oldsubsets[i]->removeReference()) {
        delete oldsubsets[i];
      }
    }
  }  // end patches

  schedulercommon_dbg << "SchedulerCommon::copyDataToNewGrid() END" << endl;
}

//______________________________________________________________________
//
void
SchedulerCommon::scheduleParticleRelocation( const LevelP&                           level,
					                                   const VarLabel*                         old_posLabel,
					                                   const vector<vector<const VarLabel*> >& old_labels,
					                                   const VarLabel*                         new_posLabel,
					                                   const vector<vector<const VarLabel*> >& new_labels,
					                                   const VarLabel*                         particleIDLabel,
					                                   const MaterialSet*                      matls,
					                                         int                               which )
{
  if (which == 1) {
    if (reloc_new_posLabel_) {
      ASSERTEQ(reloc_new_posLabel_, new_posLabel);
    }
    reloc_new_posLabel_ = new_posLabel;
    UintahParallelPort* lbp = getPort("load balancer");
    LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
    reloc1_.scheduleParticleRelocation(this, d_myworld, lb, level, old_posLabel, old_labels, new_posLabel, new_labels,
                                       particleIDLabel, matls);
    releasePort("load balancer");
  }

  if (which == 2) {
    if (reloc_new_posLabel_) {
      ASSERTEQ(reloc_new_posLabel_, new_posLabel);
    }
    reloc_new_posLabel_ = new_posLabel;
    UintahParallelPort* lbp = getPort("load balancer");
    LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
    reloc2_.scheduleParticleRelocation(this, d_myworld, lb, level, old_posLabel, old_labels, new_posLabel, new_labels,
                                       particleIDLabel, matls);
    releasePort("load balancer");
  }
}

//______________________________________________________________________
//
void
SchedulerCommon::scheduleParticleRelocation( const LevelP&                           coarsestLevelwithParticles,
                                             const VarLabel*                         old_posLabel,
                                             const vector<vector<const VarLabel*> >& old_labels,
                                             const VarLabel*                         new_posLabel,
                                             const vector<vector<const VarLabel*> >& new_labels,
                                             const VarLabel*                         particleIDLabel,
                                             const MaterialSet*                      matls )
{
  if (reloc_new_posLabel_) {
    ASSERTEQ(reloc_new_posLabel_, new_posLabel);
  }

  reloc_new_posLabel_ = new_posLabel;
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  reloc1_.scheduleParticleRelocation(this, d_myworld, lb, coarsestLevelwithParticles, old_posLabel, old_labels, new_posLabel,
                                     new_labels, particleIDLabel, matls);
  releasePort("load balancer");
}

//______________________________________________________________________
//
void
SchedulerCommon::scheduleParticleRelocation( const LevelP&                           coarsestLevelwithParticles,
                                             const VarLabel*                         posLabel,
                                             const vector<vector<const VarLabel*> >& otherLabels,
                                             const MaterialSet*                      matls )
{

  reloc_new_posLabel_ = posLabel;
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  reloc1_.scheduleParticleRelocation(this, d_myworld, lb, coarsestLevelwithParticles, posLabel, otherLabels, matls);
  releasePort("load balancer");
}

//______________________________________________________________________
//
void
SchedulerCommon::overrideVariableBehavior( string var,
                                           bool   treatAsOld,
                                           bool   copyData,
                                           bool   noScrub,
                                           bool   notCopyData,
                                           bool   noCheckpoint )
{
  // treat variable as an "old" var - will be checkpointed, copied, and only scrubbed from an OldDW
  if (treatAsOld) {
    treatAsOldVars_.insert(var);
  }

  // manually copy variable between AMR levels
  if (copyData) {
    copyDataVars_.insert(var);
    noScrubVars_.insert(var);
  }

  // ignore copying this variable between AMR levels
  if (notCopyData) {
    notCopyDataVars_.insert(var);
  }

  // set variable not to scrub (normally when needed between a normal taskgraph
  // and the regridding phase)
  if (noScrub) {
    noScrubVars_.insert(var);
  }

  // do not checkpoint this variable.
  if (noCheckpoint) {
    notCheckpointVars_.insert(var);
  }
}

//______________________________________________________________________
// output the task name and the level it's executing on, and each of the patches
void
SchedulerCommon::printTask( ostream&      out,
                            DetailedTask* task )
{
  out << left;
  out.width(70);
  out << task->getTask()->getName();

  if (task->getPatches()) {
    out << " \t on patches ";
    const PatchSubset* patches = task->getPatches();
    for (int p = 0; p < patches->size(); p++) {
      if (p != 0) {
        out << ", ";
      }
      out << patches->get(p)->getID();
    }

    if (task->getTask()->getType() != Task::OncePerProc) {
      const Level* level = getLevel(patches);
      out << "\t  L-" << level->getIndex();
    }
  }
}

//______________________________________________________________________
//  Output the task name and the level it's executing on
//  only first patch of that level
void 
SchedulerCommon::printTaskLevels( const ProcessorGroup* d_myworld,
                                        DebugStream&    out,
                                        DetailedTask*   task )
{
  if (out.active()) {
    if (task->getPatches()) {

      if (task->getTask()->getType() != Task::OncePerProc) {

        const PatchSubset* taskPatches = task->getPatches();

        const Level* level = getLevel(taskPatches);
        const Patch* firstPatch = level->getPatch(0);

        if (taskPatches->contains(firstPatch)) {

          out << d_myworld->myrank() << "   ";
          out << left;
          out.width(70);
          out << task->getTask()->getName();
          out << " \t  Patch " << firstPatch->getGridIndex() << "\t L-" << level->getIndex() << "\n";
        }
      }
    }
  }  // debugstream active
} 
