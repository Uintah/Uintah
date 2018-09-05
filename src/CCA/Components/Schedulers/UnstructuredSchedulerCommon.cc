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

#include <CCA/Components/Schedulers/UnstructuredSchedulerCommon.h>

#include <CCA/Components/Schedulers/UnstructuredDetailedTasks.h>
#include <CCA/Components/Schedulers/UnstructuredOnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/UnstructuredOnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/UnstructuredTaskGraph.h>

#include <CCA/Ports/UnstructuredApplicationInterface.h>
#include <CCA/Ports/UnstructuredDataWarehouse.h>
#include <CCA/Ports/UnstructuredLoadBalancer.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/UnstructuredPatch.h>
#include <Core/Grid/UnstructuredTask.h>
#include <Core/Grid/Variables/UnstructuredLocallyComputedPatchVarMap.h>
#include <Core/Grid/Variables/UnstructuredPerPatch.h>
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
#include <Core/Util/DOUT.hpp>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Timers/Timers.hpp>

#include <sci_defs/visit_defs.h>

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace Uintah;

namespace {
  Dout g_schedulercommon_dbg( "UnstructuredSchedulerCommon_DBG", "UnstructuredSchedulerCommon", "general debug information"  , false );
  Dout g_task_graph_compile(  "UnstructuredSchedulerCommon_TG" , "UnstructuredSchedulerCommon", "task graph compilation info", false );
}


// for calculating memory use when sci-malloc is disabled.
char* UnstructuredSchedulerCommon::start_addr = nullptr;


//______________________________________________________________________
//
UnstructuredSchedulerCommon::UnstructuredSchedulerCommon( const ProcessorGroup * myworld )
  : UintahParallelComponent( myworld )
{
  for (int i = 0; i < UnstructuredTask::TotalDWs; i++) {
    m_dwmap[i] = UnstructuredTask::InvalidDW;
  }

  // Default mapping...
  m_dwmap[UnstructuredTask::OldDW] = 0;
  m_dwmap[UnstructuredTask::NewDW] = 1;

  m_locallyComputedPatchVarMap = scinew UnstructuredLocallyComputedPatchVarMap;
}

//______________________________________________________________________
//
UnstructuredSchedulerCommon::~UnstructuredSchedulerCommon()
{
  if (m_mem_logfile) {
    delete m_mem_logfile;
  }

  // list of vars used for AMR regridding
  for (unsigned i = 0u; i < m_label_matls.size(); i++)
    for (LabelMatlMap::iterator iter = m_label_matls[i].begin(); iter != m_label_matls[i].end(); iter++)
      if (iter->second->removeReference()) {
        delete iter->second;
      }

  for (unsigned i = 0u; i < m_task_graphs.size(); i++) {
    delete m_task_graphs[i];
  }

  m_label_matls.clear();

  if (m_locallyComputedPatchVarMap) {
    delete m_locallyComputedPatchVarMap;
  }

  // Task monitoring variables.
  if (m_monitoring) {
    if (m_dummy_matl && m_dummy_matl->removeReference()) {
      delete m_dummy_matl;
    }

    // Loop through the global (0) and local (1) tasks
    for (unsigned int i = 0u; i < 2; ++i) {
      for (const auto &it : m_monitoring_tasks[i]) {
        UnstructuredVarLabel::destroy(it.second);
      }

      m_monitoring_values[i].clear();
    }
  }
}

void UnstructuredSchedulerCommon::setComponents( UintahParallelComponent *comp )
{
  UnstructuredSchedulerCommon *parent = dynamic_cast<UnstructuredSchedulerCommon*>( comp );

  attachPort( "load balancer", parent->m_loadBalancer );
  attachPort( "output",        parent->m_output );
  attachPort( "application",   parent->m_application );

  getComponents();
}

void UnstructuredSchedulerCommon::getComponents()
{
  m_loadBalancer = dynamic_cast<UnstructuredLoadBalancer*>( getPort("load balancer") );

  if( !m_loadBalancer ) {
    throw InternalError("dynamic_cast of 'm_loadBalancer' failed!", __FILE__, __LINE__);
  }

  m_output = dynamic_cast<Output*>( getPort("output") );

  if( !m_output ) {
    throw InternalError("dynamic_cast of 'm_output' failed!", __FILE__, __LINE__);
  }

  m_application = dynamic_cast<UnstructuredApplicationInterface*>( getPort("application") );

  if( !m_application ) {
    throw InternalError("dynamic_cast of 'm_application' failed!", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
void UnstructuredSchedulerCommon::releaseComponents()
{
  releasePort( "load balancer" );
  releasePort( "output" );
  releasePort( "application" );

  m_loadBalancer = nullptr;
  m_output       = nullptr;
  m_application  = nullptr;

  m_sharedState = nullptr;
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::checkMemoryUse( unsigned long & memUsed
                               , unsigned long & highwater
                               , unsigned long & maxMemUsed
                               )
{
  highwater = 0; 
  memUsed   = 0;

#if !defined(DISABLE_SCI_MALLOC)
  size_t nalloc,  sizealloc, nfree,  sizefree, nfillbin,
         nmmap, sizemmap, nmunmap, sizemunmap, highwater_alloc,  
         highwater_mmap, bytes_overhead, bytes_free, bytes_fragmented, bytes_inuse, bytes_inhunks;
  
  GetGlobalStats( DefaultAllocator(),
                  nalloc, sizealloc, nfree, sizefree,
                  nfillbin, nmmap, sizemmap, nmunmap,
                  sizemunmap, highwater_alloc, highwater_mmap,
                  bytes_overhead, bytes_free, bytes_fragmented, bytes_inuse, bytes_inhunks );
  memUsed   = sizealloc - sizefree;
  highwater = highwater_mmap;

#else

  if ( ProcessInfo::isSupported( ProcessInfo::MEM_SIZE ) ) {
    memUsed = ProcessInfo::getMemoryResident();
    // printf("1) memuse is %ld (on proc %d)\n", memuse, Uintah::Parallel::getMPIRank() );
  } else {
    memUsed = (char*)sbrk(0)-start_addr;
    // printf("2) memuse is %ld (on proc %d)\n", memuse, Uintah::Parallel::getMPIRank() );
  }
#endif

  if( memUsed > m_max_mem_used ) {
    m_max_mem_used = memUsed;
  }
  maxMemUsed = m_max_mem_used;
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::resetMaxMemValue()
{
  m_max_mem_used = 0;
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::makeTaskGraphDoc( const UnstructuredDetailedTasks * /* dt*/
                                 ,       int             rank
                                 )
{
  if (!m_emit_task_graph) {
    return;
  }

  if (!m_output->isOutputTimeStep()) {
    return;
  }

  // make sure to release this DOMDocument after finishing emitting the nodes
  m_graph_doc = ProblemSpec::createDocument("Uintah_TaskGraph");

  ProblemSpecP meta = m_graph_doc->appendChild("Meta");
  meta->appendElement("username", getenv("LOGNAME"));
  time_t t = time(nullptr);
  meta->appendElement("date", ctime(&t));

  m_graph_nodes = m_graph_doc->appendChild("Nodes");

  ProblemSpecP edgesElement = m_graph_doc->appendChild("Edges");

  for (unsigned i = 0; i < m_task_graphs.size(); i++) {
    UnstructuredDetailedTasks* dts = m_task_graphs[i]->getDetailedTasks();
    if (dts) {
      dts->emitEdges(edgesElement, rank);
    }
  }
}

//______________________________________________________________________
//
bool
UnstructuredSchedulerCommon::useInternalDeps()
{
  // Keep track of internal dependencies only if it will emit the taskgraphs (by default).
  return m_emit_task_graph;
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::emitNode( const UnstructuredDetailedTask * dtask
                         ,       double         start
                         ,       double         duration
                         ,       double         execution_duration
                         )
{  
  if ( m_graph_nodes == nullptr ) {
    return;
  }

  ProblemSpecP node = m_graph_nodes->appendChild( "node" );
  //m_graph_nodes->appendChild( node );

  node->appendElement("name", dtask->getName());
  node->appendElement("start", start);
  node->appendElement("duration", duration);

  if (execution_duration > 0) {
    node->appendElement("execution_duration", execution_duration);
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::finalizeNodes( int process /* = 0 */ )
{
  if (!m_graph_doc) {
    return;
  }

  if (m_output->isOutputTimeStep()) {
    std::string timestep_dir(m_output->getLastTimeStepOutputLocation());

    std::ostringstream fname;
    fname << "/taskgraph_" << std::setw(5) << std::setfill('0') << process << ".xml";
    std::string file_name(timestep_dir + fname.str());
    m_graph_doc->output(file_name.c_str());
  }

  // TODO: figure out why this was commented out - APH 08/06/16
  //m_graph_doc->releaseDocument();
  //m_graph_doc = nullptr;
  //m_graph_nodes = nullptr;
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::problemSetup( const ProblemSpecP     & prob_spec
                             , const SimulationStateP & state
                             )
{
  m_sharedState = state;

  m_tracking_vars_print_location = PRINT_AFTER_EXEC;

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->getWithDefault("small_messages", m_use_small_messages, true);

    if (m_use_small_messages) {
      proc0cout << "   Using small, individual MPI messages (no message combining)\n";
    }
    else {
      proc0cout << "   Using large, combined MPI messages\n";
    }

    ProblemSpecP track = params->findBlock("VarTracker");
    if (track) {
      track->require("start_time", m_tracking_start_time);
      track->require("end_time", m_tracking_end_time);
      track->getWithDefault("level", m_tracking_level, -1);
      track->getWithDefault("start_index", m_tracking_start_index, IntVector(-9, -9, -9));
      track->getWithDefault("end_index", m_tracking_end_index, IntVector(-9, -9, -9));
      track->getWithDefault("patchid", m_tracking_patch_id, -1);

      if (d_myworld->myRank() == 0) {
        std::cout << "\n";
        std::cout << "-----------------------------------------------------------\n";
        std::cout << "-- Initializing VarTracker...\n";
        std::cout << "--  Running from time " << m_tracking_start_time << " to " << m_tracking_end_time << "\n";
        std::cout << "--  for indices: " << m_tracking_start_index << " to " << m_tracking_end_index << "\n";
      }

      ProblemSpecP location = track->findBlock("locations");
      if (location) {
        m_tracking_vars_print_location = 0;
        std::map<std::string, std::string> attributes;
        location->getAttributes(attributes);
        if (attributes["before_comm"] == "true") {
          m_tracking_vars_print_location |= PRINT_BEFORE_COMM;
          proc0cout << "--  Printing variable information before communication.\n";
        }
        if (attributes["before_exec"] == "true") {
          m_tracking_vars_print_location |= PRINT_BEFORE_EXEC;
          proc0cout << "--  Printing variable information before task execution.\n";
        }
        if (attributes["after_exec"] == "true") {
          m_tracking_vars_print_location |= PRINT_AFTER_EXEC;
          proc0cout << "--  Printing variable information after task execution.\n";
        }
      }
      else {
        // "locations" not specified
        proc0cout << "--  Defaulting to printing variable information after task execution.\n";
      }

      for (ProblemSpecP var = track->findBlock("var"); var != nullptr; var = var->findNextBlock("var")) {
        std::map<std::string, std::string> attributes;
        var->getAttributes(attributes);
        std::string name = attributes["label"];
        m_tracking_vars.push_back(name);
        std::string dw = attributes["dw"];

        if (dw == "OldDW") {
          m_tracking_dws.push_back(UnstructuredTask::OldDW);
        }
        else if (dw == "NewDW") {
          m_tracking_dws.push_back(UnstructuredTask::NewDW);
        }
        else if (dw == "CoarseNewDW") {
          m_tracking_dws.push_back(UnstructuredTask::CoarseNewDW);
        }
        else if (dw == "CoarseOldDW") {
          m_tracking_dws.push_back(UnstructuredTask::CoarseOldDW);
        }
        else if (dw == "ParentOldDW") {
          m_tracking_dws.push_back(UnstructuredTask::ParentOldDW);
        }
        else if (dw == "ParentOldDW") {
          m_tracking_dws.push_back(UnstructuredTask::ParentNewDW);
        }
        else {
          // This error message most likely can go away once the .ups validation is put into place:
          printf("WARNING: Hit switch statement default... using NewDW... (This could possibly be"
                 "an error in input file specification.)\n");
          m_tracking_dws.push_back(UnstructuredTask::NewDW);
        }
        if (d_myworld->myRank() == 0) {
          std::cout << "--  Tracking variable '" << name << "' in DataWarehouse '" << dw << "'\n";
        }
      }

      for (ProblemSpecP task = track->findBlock("task"); task != nullptr; task = task->findNextBlock("task")) {
        std::map<std::string, std::string> attributes;
        task->getAttributes(attributes);
        std::string name = attributes["name"];
        m_tracking_tasks.push_back(name);
        if (d_myworld->myRank() == 0) {
          std::cout << "--  Tracking variables for specific task: " << name << "\n";
        }
      }
      if (d_myworld->myRank() == 0) {
        std::cout << "-----------------------------------------------------------\n\n";
      }
    }
    else {  // Tracking not specified
      // This 'else' won't be necessary once the .ups files are validated... but for now.
      if (d_myworld->myRank() == 0) {
        std::cout << "<VarTracker> not specified in .ups file... no variable tracking will take place.\n";
      }
    }

    // Task monitoring variables.
    ProblemSpecP taskMonitoring = params->findBlock("TaskMonitoring");
    if (taskMonitoring)
    {
      // Record the task runtime attributes on a per cell basis rather
      // than a per patch basis. Default is per patch.
      taskMonitoring->getWithDefault("per_cell", m_monitoring_per_cell, false);

      // Maps for the global tasks to be monitored.
      for (ProblemSpecP attr = taskMonitoring->findBlock("attribute");
           attr != nullptr; attr = attr->findNextBlock("attribute"))
      {
        std::string attribute = attr->getNodeValue();

        // Set the variable name AllTasks/ plus the attribute name and
        // store in a map for easy lookup by the attribute name.

        // Note: this modifided name will be needed for saving.
        m_monitoring_tasks[0][attribute] =
          UnstructuredVarLabel::create( "AllTasks/" + attribute,
                            UnstructuredPerPatch<double>::getUnstructuredTypeDescription() );

        if (d_myworld->myRank() == 0)
          std::cout << "--  Monitoring attribute " << attribute << " "
                    << "for all tasks. "
                    << "VarLabel name = 'AllTasks/" << attribute << "'"
                    << std::endl;
      }

      // Maps for the specific tasks to be monitored.
      for (ProblemSpecP task = taskMonitoring->findBlock("task");
           task != nullptr; task = task->findNextBlock("task"))
      {
        // Get the task and attribute to be monitored.
        std::map<std::string, std::string> attributes;
        task->getAttributes(attributes);
        std::string taskName  = attributes["name"];
        std::string attribute = attributes["attribute"];

        // Strip off the colons and replace with a forward slash so
        // the tasks are divided by component.
        std::string varName = taskName;
        std::size_t found = varName.find("::");
        if( found != std::string::npos)
          varName.replace(found, 2 ,"/");
        
        // Set the variable name to the task name plus the attribute
        // name and store in a map for easy lookup by the task and
        // attribute name.

        // Note: this modifided name will be needed for saving.
        m_monitoring_tasks[1][taskName + "::" + attribute] =
          UnstructuredVarLabel::create( varName + "/" + attribute,
                            UnstructuredPerPatch<double>::getUnstructuredTypeDescription() );

        if (d_myworld->myRank() == 0)
          std::cout << "--  Monitoring attribute " << attribute << " "
                    << "for task: " << taskName << ".  "
                    << "VarLabel name = '" << varName << "/" << attribute << "'"
                    << std::endl;
      }
    }

    m_monitoring = (m_monitoring_tasks[0].size() ||
                    m_monitoring_tasks[1].size() );

    if(m_monitoring)
    {
      m_dummy_matl = scinew MaterialSubset();
      m_dummy_matl->add(0);
      m_dummy_matl->addReference();
    }
  }

  // If small_messages not specified in UPS Scheduler block, still
  // report what's used
  if (m_use_small_messages) {
    proc0cout << "   Using small, individual MPI messages (no message combining)\n";
  }
  else {
    proc0cout << "   Using large, combined MPI messages\n";
  }

  m_no_scrub_vars.insert("refineFlag");
  m_no_scrub_vars.insert("refinePatchFlag");

#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( m_application->getVisIt() && !initialized ) {
    // variable 1 - Must start with the component name and have NO
    // spaces in the var name
    UnstructuredApplicationInterface::interactiveVar var;
    var.component  = "LoadBalancer";
    var.name       = "UseSmallMessages";
    var.type       = Uintah::UnstructuredTypeDescription::bool_type;
    var.value      = (void *) &m_use_small_messages;
    var.range[0]   = 0;
    var.range[1]   = 1;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_application->getUPSVars().push_back( var );

    initialized = true;
  }
#endif
}

//______________________________________________________________________
// UnstructuredhandleError()
//
// The following routine is designed to only print out a given error
// once per error type per variable.  UnstructuredhandleError is used by
// printTrackedVars() with each type of error ('errorPosition')
// condition specifically enumerated (by an integer running from 0 to 5).
//
// Returns true if the error message is displayed.
//
bool
UnstructuredhandleError(       int           errorPosition
           , const std::string & errorMessage
           , const std::string & variableName
           )
{
  static std::vector<std::map<std::string, bool> *> errorsReported(5);

  std::map<std::string, bool> * varToReportedMap = errorsReported[errorPosition];

  // TODO: this new shouldn't happen - APH 08/06/16
  if (varToReportedMap == nullptr) {
    varToReportedMap = new std::map<std::string, bool>;
    errorsReported[errorPosition] = varToReportedMap;
  }

  bool reported = (*varToReportedMap)[variableName];
  if (!reported) {
    (*varToReportedMap)[variableName] = true;
    std::cout << errorMessage << "\n";
    return true;
  }
  return false;
}

//______________________________________________________________________
//
template< class T >
void UnstructuredSchedulerCommon::printTrackedValues(       UnstructuredGridVariable<T> * var
                                        , const IntVector       & start
                                        , const IntVector       & end
                                        )
{
  std::ostringstream message;
  for (int z = start.z(); z < end.z() + 1; z++) {            // add 1 to high to include x+,y+,z+ extraCells
    for (int y = start.y(); y < end.y() + 1; y++) {

      message << d_myworld->myRank() << "  ";

      for (int x = start.x(); x < end.x() + 1; x++) {
        IntVector c(x, y, z);
        message << " " << c << ": " << (*var)[c];
      }
      message << std::endl;
    }
    message << std::endl;
  }
  DOUT(true, message.str());
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::printTrackedVars( UnstructuredDetailedTask * dtask
                                 , int            when
                                 )
{
  bool printedHeader = false;

  unsigned taskNum;
  for (taskNum = 0; taskNum < m_tracking_tasks.size(); taskNum++) {
    if (m_tracking_tasks[taskNum] == dtask->getTask()->getName())
      break;
  }

  // Print for all tasks unless one is specified (but disclude DataArchiver tasks)
  if ((taskNum == m_tracking_tasks.size() && m_tracking_tasks.size() != 0) ||
      ((std::string(dtask->getTask()->getName())).substr(0, 12) == "DataArchiver")) {
    return;
  }

  if( m_tracking_start_time > m_application->getSimTime() ||
      m_tracking_end_time   < m_application->getSimTime() ) {
    return;
  }

  for (int i = 0; i < static_cast<int>(m_tracking_vars.size()); i++) {
    bool printedVarName = false;

    // that DW may not have been mapped....
    if (dtask->getTask()->mapDataWarehouse(m_tracking_dws[i]) < 0 ||
        dtask->getTask()->mapDataWarehouse(m_tracking_dws[i]) >= (int)m_dws.size()) {

      std::ostringstream mesg;
      mesg << "WARNING: VarTracker: Not printing requested variable (" << m_tracking_vars[i] << ") DW is out of range.\n";

      UnstructuredhandleError(0, mesg.str(), m_tracking_vars[i]);

      continue;
    }

    UnstructuredOnDemandDataWarehouseP dw = m_dws[dtask->getTask()->mapDataWarehouse(m_tracking_dws[i])];

    if (dw == nullptr) { // old on initialization timestep
      continue;
    }

    // Get the level here, as the grid can be different between the old and new DW
    const UnstructuredGrid* grid = dw->getGrid();

    int levelnum;

    if (m_tracking_level == -1) {
      levelnum = grid->numLevels() - 1;
    }
    else {
      levelnum = m_tracking_level;
      if (levelnum >= grid->numLevels()) {
        continue;
      }
    }

    const UnstructuredLevelP level = grid->getLevel(levelnum);
    const UnstructuredVarLabel* label = UnstructuredVarLabel::find(m_tracking_vars[i]);

    std::cout.precision(16);

    if (!label) {
      std::ostringstream mesg;
      mesg << "WARNING: VarTracker: Not printing requested variable (" << m_tracking_vars[i]
           << ") because the label could not be found.\n";
      UnstructuredhandleError(1, mesg.str(), m_tracking_vars[i]);
      continue;
    }

    const UnstructuredPatchSubset* patches = dtask->getPatches();

    // a once-per-proc task is liable to have multiple levels, and thus calls to getLevel(patches) will fail
    if (dtask->getTask()->getType() != UnstructuredTask::OncePerProc && dtask->getTask()->getType() != UnstructuredTask::Hypre && (!patches || getLevel(patches)->getIndex() != levelnum)) {
      std::ostringstream mesg;
      mesg << "WARNING: VarTracker: Not printing requested variable (" << m_tracking_vars[i] << ") because patch is non-standard.\n";
      UnstructuredhandleError(2, mesg.str(), m_tracking_vars[i]);
      continue;
    }

    //__________________________________
    //
    for (int p = 0; patches && p < patches->size(); p++) {

      const UnstructuredPatch* patch = patches->get(p);
      if (m_tracking_patch_id != -1 && m_tracking_patch_id != patch->getID()) {
        continue;
      }

      // Don't print ghost patches (dw->get will yell at you).
      if ((m_tracking_dws[i] == UnstructuredTask::OldDW && m_loadBalancer->getOldProcessorAssignment(patch) != d_myworld->myRank()) ||
          (m_tracking_dws[i] == UnstructuredTask::NewDW && m_loadBalancer->getPatchwiseProcessorAssignment(patch) != d_myworld->myRank())) {
        continue;
      }

      const UnstructuredTypeDescription* td = label->typeDescription();
      UnstructuredPatch::VariableBasis basis = patch->translateTypeToBasis(td->getUnstructuredType(), false);

      IntVector start = Max(patch->getExtraLowIndex( basis, IntVector(0, 0, 0)), m_tracking_start_index);
      IntVector end   = Min(patch->getExtraHighIndex(basis, IntVector(0, 0, 0)), m_tracking_end_index);

      // Loop over matls too...
      for (int m = 0; m < m_sharedState->getNumMatls(); m++) {

        if (!dw->exists(label, m, patch)) {
          std::ostringstream mesg;
          mesg << "WARNING: VarTracker: Not printing requested variable (" << m_tracking_vars[i]
               << ") because it does not exist in DW.\n" << "            Patch is: " << *patch << "\n";
          if (UnstructuredhandleError(3, mesg.str(), m_tracking_vars[i])) {
          }
          continue;
        }

        if (!(start.x() < end.x() && start.y() < end.y() && start.z() < end.z())) {
          continue;
        }

        const UnstructuredTypeDescription::UnstructuredType subType = td->getSubType()->getUnstructuredType();
        if (subType != UnstructuredTypeDescription::double_type && subType != UnstructuredTypeDescription::int_type && subType != UnstructuredTypeDescription::Vector) {

          // Only allow *Variable<double>, *Variable<int> and *Variable<Vector> for now.
          std::ostringstream mesg;
          mesg << "WARNING: VarTracker: Not printing requested variable (" << m_tracking_vars[i]
               << ") because its type is not supported:\n" << "             " << td->getName() << "\n";

          UnstructuredhandleError(4, mesg.str(), m_tracking_vars[i]);
          continue;
        }

        // pending the task that allocates the var, we may not have allocated it yet
        UnstructuredGridVariableBase* v;
        switch (td->getUnstructuredType()) {
          case UnstructuredTypeDescription::CCVariable :
          case UnstructuredTypeDescription::NCVariable :
          case UnstructuredTypeDescription::SFCXVariable :
          case UnstructuredTypeDescription::SFCYVariable :
          case UnstructuredTypeDescription::SFCZVariable :
            v = dynamic_cast<UnstructuredGridVariableBase*>(dw->d_varDB.get(label, m, patch));
            break;
          default :
            throw InternalError("Cannot track var type of non-grid-type", __FILE__, __LINE__);
            break;
        }

        start = Max(start, v->getLow());
        end = Min(end, v->getHigh());

        if (!(start.x() < end.x() && start.y() < end.y() && start.z() < end.z())) {
          continue;
        }

        if (!printedHeader) {
          std::string location;
          switch (when) {
            case PRINT_BEFORE_COMM :
              location = " before communication of ";
              break;
            case PRINT_BEFORE_EXEC :
              location = " before execution of ";
              break;
            case PRINT_AFTER_EXEC :
              location = " after execution of ";
              break;
          }
          std::cout << d_myworld->myRank() << location << *dtask << std::endl;
          printedHeader = true;
        }

        if (!printedVarName) {
          std::cout << d_myworld->myRank() << "  Variable: " << m_tracking_vars[i] << ", DW " << dw->getID() << ", Patch "
                    << patch->getID() << ", Matl " << m << std::endl;
        }

        switch (subType) {
          case UnstructuredTypeDescription::double_type : {
            UnstructuredGridVariable<double>* var = dynamic_cast<UnstructuredGridVariable<double>*>(v);
            printTrackedValues<double>(var, start, end);
          }
            break;
          case UnstructuredTypeDescription::int_type : {
            UnstructuredGridVariable<int>* var = dynamic_cast<UnstructuredGridVariable<int>*>(v);
            printTrackedValues<int>(var, start, end);
          }
            break;
          case UnstructuredTypeDescription::Vector : {
            UnstructuredGridVariable<Vector>* var = dynamic_cast<UnstructuredGridVariable<Vector>*>(v);
            printTrackedValues<Vector>(var, start, end);
          }
            break;
          default :
            break;
        }  // end case variable type
      }  // end for materials loop
    }  // end for patches loop
  }  // end for i : trackingVars.size()
}  // end printTrackedVars()

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::addTaskGraph( UnstructuredScheduler::tgType type
                             , int               index
                             )
{
  UnstructuredTaskGraph* tg = scinew UnstructuredTaskGraph(this, m_sharedState, d_myworld, type, index);
  tg->initialize();
  m_task_graphs.push_back(tg);
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::addTask(       UnstructuredTask        * task
                        , const UnstructuredPatchSet    * patches
                        , const MaterialSet * matls
                        , const int           tg_num /* = -1 */
                        )
{
  // Save the DW map
  task->setMapping(m_dwmap);

  DOUT(g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << " adding Task: " << task->getName()
                                      << ",  # patches: "    << (patches ? patches->size() : 0)
                                      << ",    # matls: "    << (matls ? matls->size() : 0)
                                      << ", task-graph: "    << ((tg_num < 0) ? (m_is_init_timestep ? "init-tg" : "all") : std::to_string(tg_num)));

  // use std::shared_ptr as task pointers may be added to all task graphs - automagic cleanup
  std::shared_ptr<UnstructuredTask> task_sp(task);

  // default case for normal tasks
  addTask(task_sp, patches, matls, tg_num);

  // separate out the standard from the distal ghost cell requirements - for loadbalancer
  // This isn't anything fancy, and could be expanded/modified down the road.
  // It just gets a max ghost cell extent for anything less than MAX_HALO_DEPTH, and
  // another max ghost cell extent for anything >= MAX_HALO_DEPTH.  The idea is that later
  // we will create two neighborhoods with max extents for each as determined here.
  for (auto dep = task->getRequires(); dep != nullptr; dep = dep->m_next) {
    if (dep->m_num_ghost_cells >= MAX_HALO_DEPTH) {
      if (dep->m_num_ghost_cells > this->m_max_distal_ghost_cells) {
        this->m_max_distal_ghost_cells = dep->m_num_ghost_cells;
      }
    }
    else {
      if (dep->m_num_ghost_cells > this->m_max_ghost_cells) {
        this->m_max_ghost_cells = dep->m_num_ghost_cells;
      }
    }
  }

  if (task->m_max_level_offset > this->m_max_level_offset) {
    this->m_max_level_offset = task->m_max_level_offset;
  }

  // add to init-requires.  These are the vars which require from the OldDW that we'll
  // need for checkpointing, switching, and the like.
  // In the case of treatAsOld Vars, we handle them because something external to the taskgraph
  // needs it that way (i.e., Regridding on a restart requires checkpointed refineFlags).
  for (auto dep = task->getRequires(); dep != nullptr; dep = dep->m_next) {
    if (isOldDW(dep->mapDataWarehouse()) || m_treat_as_old_vars.find(dep->m_var->getName()) != m_treat_as_old_vars.end()) {
      m_init_requires.push_back(dep);
      m_init_required_vars.insert(dep->m_var);
    }
  }

  // for the treat-as-old vars, go through the computes and add them.
  // we can (probably) safely assume that we'll avoid duplicates, since if they were inserted 
  // in the above, they wouldn't need to be marked as such
  for (auto dep = task->getComputes(); dep != nullptr; dep = dep->m_next) {
    m_computed_vars.insert(dep->m_var);

    if (m_treat_as_old_vars.find(dep->m_var->getName()) != m_treat_as_old_vars.end()) {
      m_init_requires.push_back(dep);
      m_init_required_vars.insert(dep->m_var);
    }
  }

  //__________________________________
  // create reduction task if computes included one or more reduction vars
  for (auto dep = task->getComputes(); dep != nullptr; dep = dep->m_next) {
    
    if (dep->m_var->typeDescription()->isReductionVariable()) {
      int levelidx = dep->m_reduction_level ? dep->m_reduction_level->getIndex() : -1;
      int dw = dep->mapDataWarehouse();

      if (dep->m_var->allowsMultipleComputes()) {
        DOUT( g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << " Skipping Reduction task for multi compute variable: "
                                             << dep->m_var->getName() << " on level " << levelidx << ", DW " << dw);
        continue;
      }

      DOUT( g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << " Creating Reduction task for variable: "
                                           << dep->m_var->getName() << " on level " << levelidx << ", DW " << dw);

      std::ostringstream taskname;
      taskname << "Reduction: " << dep->m_var->getName() << ", level " << levelidx << ", dw " << dw;

      UnstructuredTask* reduction_task = scinew UnstructuredTask(taskname.str(), UnstructuredTask::Reduction);
      
      int dwmap[UnstructuredTask::TotalDWs];
      
      for (int i = 0; i < UnstructuredTask::TotalDWs; i++) {
        dwmap[i] = UnstructuredTask::InvalidDW;
      }

      dwmap[UnstructuredTask::OldDW] = UnstructuredTask::NoDW;
      dwmap[UnstructuredTask::NewDW] = dw;
      reduction_task->setMapping(dwmap);

      int matlIdx = -1;
      if (dep->m_matls != nullptr) {
        reduction_task->modifies(dep->m_var, dep->m_reduction_level, dep->m_matls, UnstructuredTask::OutOfDomain);
        for (int i = 0; i < dep->m_matls->size(); i++) {
          matlIdx = dep->m_matls->get(i);
          const UnstructuredDataWarehouse* const_dw = get_dw(dw);
          UnstructuredVarLabelMatl<UnstructuredLevel,UnstructuredDataWarehouse> key(dep->m_var, matlIdx, dep->m_reduction_level, const_dw);

          // For reduction variables there may be multiple computes
          // each of which will create reduction task. The last
          // reduction task should be kept. This is because the tasks
          // do not get sorted.
          if( m_reduction_tasks.find(key) == m_reduction_tasks.end() )
          {
            DOUT( g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << " Excluding previous reduction task for variable: " << dep->m_var->getName() << " on level " << levelidx << ", DW " << dw << " dep->m_reduction_level " << dep->m_reduction_level << " material index " << matlIdx );
          }
          m_reduction_tasks[key] = reduction_task;

        }
      }
      else {
        for (int m = 0; m < task->getMaterialSet()->size(); m++) {
          reduction_task->modifies(dep->m_var, dep->m_reduction_level, task->getMaterialSet()->getSubset(m), UnstructuredTask::OutOfDomain);
          for (int i = 0; i < task->getMaterialSet()->getSubset(m)->size(); ++i) {
            matlIdx = task->getMaterialSet()->getSubset(m)->get(i);
            const UnstructuredDataWarehouse* const_dw = get_dw(dw);
            UnstructuredVarLabelMatl<UnstructuredLevel,UnstructuredDataWarehouse> key(dep->m_var, matlIdx, dep->m_reduction_level, const_dw);

            // For reduction variables there may be multiple computes
            // each of which will create reduction task. The last
            // reduction task should be kept. This is because the
            // tasks do not get sorted.
            if( m_reduction_tasks.find(key) == m_reduction_tasks.end() )
            {
              DOUT( g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << " Excluding previous reduction task for variable: " << dep->m_var->getName() << " on level " << levelidx << ", DW " << dw << " dep->m_reduction_level " << dep->m_reduction_level << " material index " << matlIdx );
            }

            m_reduction_tasks[key] = reduction_task;
          }
        }
      }

      // use std::shared_ptr as task pointers may be added to all task graphs - automagic cleanup
      std::shared_ptr<UnstructuredTask> reduction_task_sp(reduction_task);

      // add reduction task to the task graphs
      addTask(reduction_task_sp, nullptr, task->getMaterialSet(), tg_num);
    }
  }
}

//______________________________________________________________________
//
void UnstructuredSchedulerCommon::addTask(       std::shared_ptr<UnstructuredTask>   task
                             , const UnstructuredPatchSet              * patches
                             , const MaterialSet           * matls
                             , const int                     tg_num
                             )
{
  // During initialization, use only one task graph
  if (m_is_init_timestep) {
    m_task_graphs[m_task_graphs.size() - 1]->addTask(task, patches, matls);
    m_num_tasks++;
  }
  else {
    // add it to all "Normal" task graphs (default value == -1)
    if (tg_num < 0) {
      for (int i = 0; i < m_num_task_graphs; ++i) {
        m_task_graphs[i]->addTask(task, patches, matls);
        m_num_tasks++;
      }
    }
    // otherwise, add this task to a specific task graph
    else {
      m_task_graphs[tg_num]->addTask(task, patches, matls);
      m_num_tasks++;
    }
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::initialize( int numOldDW /* = 1 */
                           , int numNewDW /* = 1 */
                           )
{
  // doesn't really do anything except initialize/clear the taskgraph
  //   if the default parameter values are used
  int numDW = numOldDW + numNewDW;
  int oldnum = (int)m_dws.size();

  // in AMR cases we will often need to move from many new DWs to one.  In those cases, move the last NewDW to be the next new one.
  if (oldnum - m_num_old_dws > 1) {
    m_dws[numDW - 1] = m_dws[oldnum - 1];
  }

  // Clear out the data warehouse so that memory will be freed
  for (int i = numDW; i < oldnum; i++) {
    m_dws[i] = 0;
  }

  m_dws.resize(numDW);
  for (; oldnum < numDW; oldnum++) {
    m_dws[oldnum] = 0;
  }

  m_num_old_dws = numOldDW;

  // clear the taskgraphs, and set the first one
  for (unsigned i = 0; i < m_task_graphs.size(); i++) {
    delete m_task_graphs[i];
  }

  m_task_graphs.clear();

  m_init_requires.clear();
  m_init_required_vars.clear();
  m_computed_vars.clear();

  m_num_tasks               = 0;
  m_max_ghost_cells         = 0;
  m_max_distal_ghost_cells  = 0;
  m_max_level_offset        = 0;

  m_reduction_tasks.clear();

  bool is_init = m_is_init_timestep || m_is_restart_init_timestep;
  size_t num_task_graphs = (is_init) ? 1 : m_num_task_graphs;
  for (size_t i = 0; i < num_task_graphs; ++i) {
    addTaskGraph(NormalTaskGraph, i);
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::setParentDWs( UnstructuredDataWarehouse * parent_old_dw
                             , UnstructuredDataWarehouse * parent_new_dw
                             )
{
  UnstructuredOnDemandDataWarehouse* pold = dynamic_cast<UnstructuredOnDemandDataWarehouse*>(parent_old_dw);
  UnstructuredOnDemandDataWarehouse* pnew = dynamic_cast<UnstructuredOnDemandDataWarehouse*>(parent_new_dw);

  if (parent_old_dw && parent_new_dw) {

    ASSERT(pold != nullptr);
    ASSERT(pnew != nullptr);
    ASSERT(m_num_old_dws > 2);

    m_dws[0] = pold;
    m_dws[1] = pnew;
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::clearMappings()
{
  for (int i = 0; i < UnstructuredTask::TotalDWs; i++) {
    m_dwmap[i] = -1;
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::mapDataWarehouse( UnstructuredTask::WhichDW which
                                 , int           dwTag
                                 )
{
  ASSERTRANGE(which, 0, UnstructuredTask::TotalDWs);
  ASSERTRANGE(dwTag, 0, static_cast<int>(m_dws.size()));

  m_dwmap[which] = dwTag;
}

//______________________________________________________________________
//
UnstructuredDataWarehouse*
UnstructuredSchedulerCommon::get_dw( int idx )
{
  ASSERTRANGE(idx, 0, static_cast<int>(m_dws.size()));

  if( 0 <= idx &&  idx < static_cast<int>(m_dws.size()) )
    return m_dws[idx].get_rep();
  else
    return nullptr;
}

//______________________________________________________________________
//
UnstructuredDataWarehouse*
UnstructuredSchedulerCommon::getLastDW()
{
  return get_dw(static_cast<int>(m_dws.size()) - 1);
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::advanceDataWarehouse( const UnstructuredGridP & grid
                                     ,       bool    initialization /* = false */
                                     )
{
  DOUT(g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << " advanceDataWarehouse, numDWs = " << m_dws.size());

  ASSERT(m_dws.size() >= 2);

  // The last becomes last old, and the rest are new
  m_dws[m_num_old_dws - 1] = m_dws[m_dws.size() - 1];

  if( m_dws.size() == 2 && m_dws[0] == nullptr ) {
    // first datawarehouse -- indicate that it is the "initialization" dw.
    int generation = m_generation++;
    m_dws[1] = scinew UnstructuredOnDemandDataWarehouse(d_myworld, this, generation, grid, true /* initialization dw */);
  }
  else {
    for (int i = m_num_old_dws; i < static_cast<int>(m_dws.size()); i++) {
      // in AMR initial cases, you can still be in initialization when you advance again
      replaceDataWarehouse(i, grid, initialization);
    }
  }
}

//______________________________________________________________________
//
void 
UnstructuredSchedulerCommon::fillDataWarehouses( const UnstructuredGridP & grid )
{
  for (int i = m_num_old_dws; i < static_cast<int>(m_dws.size()); i++) {
    if (!m_dws[i]) {
      replaceDataWarehouse(i, grid);
    }
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::replaceDataWarehouse(       int     index
                                     , const UnstructuredGridP & grid
                                     ,       bool    initialization /* = false */
                                     )
{
  m_dws[index] = scinew UnstructuredOnDemandDataWarehouse(d_myworld, this, m_generation++, grid, initialization);

  if (initialization) {
    return;
  }
  
  for (unsigned i = 0; i < m_task_graphs.size(); i++) {
    UnstructuredDetailedTasks* dts = m_task_graphs[i]->getDetailedTasks();
    if (dts) {
      dts->copyoutDWKeyDatabase(m_dws[index]);
    }
  }
  m_dws[index]->doReserve();
}

//______________________________________________________________________
//
void 
UnstructuredSchedulerCommon::setRestartable( bool restartable )
{
  m_restartable = restartable;
}

//______________________________________________________________________
//
const std::vector<const UnstructuredPatch*>*
UnstructuredSchedulerCommon::getSuperPatchExtents( const UnstructuredVarLabel         * label
                                     ,       int                matlIndex
                                     , const UnstructuredPatch            * patch
                                     ,       Ghost::GhostType   requestedGType
                                     ,       int                requestedNumGCells
                                     ,       IntVector        & requiredLow
                                     ,       IntVector        & requiredHigh
                                     ,       IntVector        & requestedLow
                                     ,       IntVector        & requestedHigh
                                     ) const
{
  const SuperPatch* connectedPatchGroup = m_locallyComputedPatchVarMap->getConnectedPatchGroup(patch);

  if (connectedPatchGroup == nullptr) {
    return nullptr;
  }

  SuperPatch::Region requestedExtents = connectedPatchGroup->getRegion();
  SuperPatch::Region requiredExtents = connectedPatchGroup->getRegion();

  // expand to cover the entire connected patch group
  for (unsigned int i = 0; i < connectedPatchGroup->getBoxes().size(); i++) {
    // get the minimum extents containing both the expected ghost cells
    // to be needed and the given ghost cells.
    const UnstructuredPatch* memberPatch = connectedPatchGroup->getBoxes()[i];

    UnstructuredPatch::VariableBasis basis = UnstructuredPatch::translateTypeToBasis(label->typeDescription()->getUnstructuredType(), true);

    IntVector lowOffset  = IntVector(0, 0, 0);
    IntVector highOffset = IntVector(0, 0, 0);

    //set requiredLow and requiredHigh as extents without ghost cells
    memberPatch->computeExtents(basis, label->getBoundaryLayer(), lowOffset, highOffset, requiredLow, requiredHigh);

    //compute ghost cell offsets
    UnstructuredPatch::getGhostOffsets(basis, requestedGType, requestedNumGCells, lowOffset, highOffset);

    //set requestedLow and requestedHigh as extents with ghost cells
    memberPatch->computeExtents(basis, label->getBoundaryLayer(), lowOffset, highOffset, requestedLow, requestedHigh);

    SuperPatch::Region requiredRegion = SuperPatch::Region(requiredLow, requiredHigh);
    requiredExtents = requiredExtents.enclosingRegion(requiredRegion);
    SuperPatch::Region requestedRegion = SuperPatch::Region(requestedLow, requestedHigh);
    requestedExtents = requestedExtents.enclosingRegion(requestedRegion);

    ASSERT(memberPatch == patch);
  }

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
UnstructuredSchedulerCommon::logMemoryUse()
{
  if (!m_mem_logfile) {
    std::ostringstream fname;
    fname << "uintah_memuse.log.p" << std::setw(5) << std::setfill('0') << d_myworld->myRank() << "." << d_myworld->nRanks();
    m_mem_logfile = scinew std::ofstream(fname.str().c_str());
    if (!m_mem_logfile) {
      std::cerr << "Error opening file: " << fname.str() << '\n';
    }
  }

  *m_mem_logfile << '\n';
  unsigned long total = 0;

  for (int i = 0; i < (int)m_dws.size(); i++) {
    char* name;
    if (i == 0) {
      name = const_cast<char*>("OldDW");
    } else if (i == (int)m_dws.size() - 1) {
      name = const_cast<char*>("NewDW");
    } else {
      name = const_cast<char*>("IntermediateDW");
    }

    if (m_dws[i]) {
      m_dws[i]->logMemoryUse(*m_mem_logfile, total, name);
    }

  }

  for (unsigned i = 0; i < m_task_graphs.size(); i++) {
    UnstructuredDetailedTasks* dts = m_task_graphs[i]->getDetailedTasks();
    if (dts) {
      dts->logMemoryUse(*m_mem_logfile, total, "Taskgraph");
    }
  }

  *m_mem_logfile << "Total: " << total << '\n';
  m_mem_logfile->flush();
}

//______________________________________________________________________
//
// Makes and returns a map that maps strings to VarLabels of
// that name and a list of material indices for which that
// variable is valid (according to d_allcomps in graph).
UnstructuredScheduler::VarLabelMaterialMap*
UnstructuredSchedulerCommon::makeVarLabelMaterialMap()
{
  VarLabelMaterialMap* result = scinew VarLabelMaterialMap;
  for (unsigned i = 0; i < m_task_graphs.size(); i++) {
    m_task_graphs[i]->makeVarLabelMaterialMap(result);
  }
  return result;
}

//______________________________________________________________________
//     
void
UnstructuredSchedulerCommon::doEmitTaskGraphDocs()
{
  m_emit_task_graph = true;
}

//______________________________________________________________________
//

void
UnstructuredSchedulerCommon::compile()
{
  UnstructuredGridP grid = const_cast<UnstructuredGrid*>(getLastDW()->getGrid());
  UnstructuredGridP oldGrid = nullptr;

  if (m_dws[0]) {
    oldGrid = const_cast<UnstructuredGrid*>(get_dw(0)->getGrid());
  }
  
  if (m_num_tasks > 0) {

    DOUT(g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << " UnstructuredSchedulerCommon starting compile");

    auto num_task_graphs = m_task_graphs.size();
    for (auto i = 0u; i < num_task_graphs; i++) {
      if (num_task_graphs > 1) {
        DOUT(g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << "  Compiling task graph: " << i+1 << " of " << m_task_graphs.size() << " with " << m_num_tasks << " tasks.");
      }

      Timers::Simple tg_compile_timer;

      // do we have tasks with halo requirements > MAX_HALO_DEPTH? Determined in UnstructuredSchedulerCommon::addTask
      const bool has_distal_reqs = ( (m_task_graphs[i]->getIndex() > 0) && (m_max_distal_ghost_cells != 0) );


      //--------------------------------------------------------------------------------------------
      // NOTE: this single call is where all the TG compilation complexity arises
      //--------------------------------------------------------------------------------------------
      m_task_graphs[i]->createDetailedTasks( useInternalDeps(), grid, oldGrid, has_distal_reqs );
      //--------------------------------------------------------------------------------------------


      double compile_time = tg_compile_timer().seconds();

      DOUT(g_task_graph_compile, "Rank-" << std::left << std::setw(5) << d_myworld->myRank() << " time to compile TG-" << std::setw(4)
                                         << (m_is_init_timestep ? "INIT" : std::to_string(m_task_graphs[i]->getIndex())) << ": " << compile_time << " (sec)");
    }

    // polymorphically check scheduler at runtime, that all ranks are executing the same size TG (excluding spatial tasks)
    verifyChecksum();

    DOUT(g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << " UnstructuredSchedulerCommon finished compile");

  } else {
    // NOTE: this was added with scheduleRestartInititalize() support (for empty TGs)
    //  Even when numTasks_ <= 0, the code below executed and did nothing worthwhile... seemingly
    //  Shouldn't need to do this work without tasks though -APH 03/12/15
    return; // no tasks and nothing to do
  }

  m_locallyComputedPatchVarMap->reset();
  
  // TODO: which of the two cases should we be using and why do both exist - APH 02/23/18
  //       looks like this sets up superbox and superpatch sets (m_locallyComputedPatchVarMap)
#if 1
  for (int i = 0; i < grid->numLevels(); i++) {
    const UnstructuredPatchSubset* patches = m_loadBalancer->getPerProcessorPatchSet(grid->getLevel(i))->getSubset(d_myworld->myRank());

    if (patches->size() > 0) {
      m_locallyComputedPatchVarMap->addComputedPatchSet(patches);
    }
  }

#else
  for (auto i = 0u; i < m_task_graphs.size(); i++) {
    UnstructuredDetailedTasks* dts = m_task_graphs[i]->getDetailedTasks();

    if (dts != nullptr) {
      // figure out the locally computed patches for each variable.
      for (auto i = 0; i < dts->numLocalTasks(); i++) {
        const UnstructuredDetailedTask* dtask = dts->localTask(i);

        for(const UnstructuredTask::Dependency* comp = dtask->getTask()->getComputes(); comp != nullptr; comp = comp->m_next) {

          if (comp->m_var->typeDescription()->getUnstructuredType() != UnstructuredTypeDescription::UnstructuredReductionVariable) {
            constHandle<UnstructuredPatchSubset> patches = comp->getPatchesUnderDomain(dtask->getPatches());
            m_locallyComputedPatchVarMap->addComputedPatchSet(patches.get_rep());
          }
        }
      }
    }
  }
#endif

  for (auto dw = 0u; dw < m_dws.size(); dw++) {
    if (m_dws[dw].get_rep()) {
      for (auto i = 0u; i < m_task_graphs.size(); i++) {
        UnstructuredDetailedTasks* dts = m_task_graphs[i]->getDetailedTasks();
        dts->copyoutDWKeyDatabase(m_dws[dw]);
      }
      m_dws[dw]->doReserve();
    }
  }

  m_locallyComputedPatchVarMap->makeGroups();
}

//______________________________________________________________________
//
bool
UnstructuredSchedulerCommon::isOldDW( int idx ) const
{
  ASSERTRANGE(idx, 0, static_cast<int>(m_dws.size()));
  return idx < m_num_old_dws;
}

//______________________________________________________________________
//
bool
UnstructuredSchedulerCommon::isNewDW( int idx ) const
{
  ASSERTRANGE(idx, 0, static_cast<int>(m_dws.size()));
  return idx >= m_num_old_dws;
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::finalizeTimestep()
{
  finalizeNodes(d_myworld->myRank());

  for (unsigned int i = m_num_old_dws; i < m_dws.size(); i++) {
    m_dws[i]->finalize();
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::scheduleAndDoDataCopy( const UnstructuredGridP & grid )
{
  Timers::Simple timer;
  timer.start();

  // TODO - use the current initReqs and push them back, instead of doing this...
  // clear the old list of vars and matls
  for (unsigned i = 0; i < m_label_matls.size(); i++) {
    for (LabelMatlMap::iterator iter = m_label_matls[i].begin(); iter != m_label_matls[i].end(); iter++) {
      if (iter->second->removeReference()) {
        delete iter->second;
      }
    }
  }

  m_label_matls.clear();
  m_label_matls.resize(grid->numLevels());

  // produce a map from all tasks' requires from the Old DW.  Store the varlabel and matls
  // TODO - only do this ONCE.
  for (unsigned t = 0; t < m_task_graphs.size(); t++) {
    UnstructuredTaskGraph* tg = m_task_graphs[t];
    for (int i = 0; i < tg->getNumTasks(); i++) {
      UnstructuredTask* task = tg->getTask(i);
      if (task->getType() == UnstructuredTask::Output) {
        continue;
      }

      for (const UnstructuredTask::Dependency* dep = task->getRequires(); dep != 0; dep = dep->m_next) {
        bool copyThisVar = dep->m_whichdw == UnstructuredTask::OldDW;
        // override to manually copy a var
        if (!copyThisVar) {
          if (m_copy_data_vars.find(dep->m_var->getName()) != m_copy_data_vars.end()) {
            copyThisVar = true;
          }
        }

        // Overide the logic above.  There are PerPatch variables that cannot/shouldn't be copied to the new grid,
        // for example PerPatch<FileInfo>.
        if (m_no_copy_data_vars.count(dep->m_var->getName()) > 0) {
          copyThisVar = false;
        }

        if (copyThisVar) {
          if (dep->m_var->typeDescription()->getUnstructuredType() == UnstructuredTypeDescription::UnstructuredReductionVariable) {
            // we will take care of reduction variables in a different section
            continue;
          }

          // check the level on the case where variables are only computed on certain levels
          const UnstructuredPatchSet* ps = task->getPatchSet();
          int level = -1;
          if (dep->m_patches) {        // just in case the task is over multiple levels...
            level = getLevel(dep->m_patches)->getIndex();
          } else if (ps) {
            level = getLevel(ps)->getIndex();
          }

          // we don't want data with an invalid level, or requiring from a different level (remember, we are
          // using an old task graph).  That will be copied later (and chances are, it's to modify anyway).
          if (level == -1 || level > grid->numLevels() - 1 ||
              dep->m_patches_dom == UnstructuredTask::CoarseLevel      ||
              dep->m_patches_dom == UnstructuredTask::FineLevel) {
            continue;
          }

          const MaterialSubset * matSubset = (dep->m_matls != 0) ? dep->m_matls : dep->m_task->getMaterialSet()->getUnion();

          // if var was already found, make a union of the materials
          MaterialSubset* matls = scinew MaterialSubset(matSubset->getVector());
          matls->addReference();

          MaterialSubset* union_matls;
          union_matls = m_label_matls[level][dep->m_var];

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
          m_label_matls[level][dep->m_var] = matls;
        }
      }
    }
  }

  this->initialize(1, 1);
  this->advanceDataWarehouse(grid, true);
  this->clearMappings();
  this->mapDataWarehouse(UnstructuredTask::OldDW, 0);
  this->mapDataWarehouse(UnstructuredTask::NewDW, 1);
  this->mapDataWarehouse(UnstructuredTask::CoarseOldDW, 0);
  this->mapDataWarehouse(UnstructuredTask::CoarseNewDW, 1);

  UnstructuredDataWarehouse* oldDataWarehouse = this->get_dw(0);
  UnstructuredDataWarehouse* newDataWarehouse = this->getLastDW();

  oldDataWarehouse->setScrubbing(UnstructuredDataWarehouse::ScrubNone);
  newDataWarehouse->setScrubbing(UnstructuredDataWarehouse::ScrubNone);

  const UnstructuredGrid * oldGrid = oldDataWarehouse->getGrid();

  std::vector<UnstructuredTask*> dataTasks;
  std::vector<Handle<UnstructuredPatchSet> > refinePatchSets(grid->numLevels(), (UnstructuredPatchSet*)0);
  std::vector<Handle<UnstructuredPatchSet> > copyPatchSets(grid->numLevels(), (UnstructuredPatchSet*)0);
  UnstructuredSchedulerP sched(dynamic_cast<UnstructuredScheduler*>(this));

  m_is_copy_data_timestep = true;

  for (int L = 0; L < grid->numLevels(); L++) {
    UnstructuredLevelP newLevel = grid->getLevel(L);

    if (L > 0) {

      if (L >= oldGrid->numLevels()) {
        // new level - refine everywhere
        refinePatchSets[L] = const_cast<UnstructuredPatchSet*>(newLevel->eachPatch());
        copyPatchSets[L] = scinew UnstructuredPatchSet;
      }

      // Find patches with new space - but temporarily, refine everywhere... 
      else if (L < oldGrid->numLevels()) {
        refinePatchSets[L] = scinew UnstructuredPatchSet;
        copyPatchSets[L] = scinew UnstructuredPatchSet;

        std::vector<int> myPatchIDs;
        UnstructuredLevelP oldLevel = oldDataWarehouse->getGrid()->getLevel(L);

        // Go through the patches, and find if there are patches that
        // weren't entirely covered by patches on the old grid, and
        // interpolate them.  then after, copy the data, and if
        // necessary, overwrite interpolated data
        const UnstructuredPatchSubset *ps = m_loadBalancer->getPerProcessorPatchSet(newLevel)->getSubset(d_myworld->myRank());

        // for each patch I own
        for (int p = 0; p < ps->size(); p++) {
          const UnstructuredPatch *newPatch = ps->get(p);

          // get the low/high for what we'll need to get
          IntVector lowIndex, highIndex;
          //newPatch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0), Ghost::None, 0, lowIndex, highIndex);
          lowIndex = newPatch->getCellLowIndex();
          highIndex = newPatch->getCellHighIndex();

          // find if area on the new patch was not covered by the old patches
          IntVector dist = highIndex - lowIndex;
          int totalCells = dist.x() * dist.y() * dist.z();
          int sum = 0;
          UnstructuredPatch::selectType oldPatches;
          oldLevel->selectPatches(lowIndex, highIndex, oldPatches);

          //compute volume of overlapping regions
          for (unsigned int old = 0; old < oldPatches.size(); old++) {

            const UnstructuredPatch* oldPatch = oldPatches[old];
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
            } else {
              refinePatchSets[L]->add(newPatch);
            }
          } else {
            if (!Uintah::Parallel::usingMPI()) {
              copyPatchSets[L]->add(newPatch);
            }
          }
        }  // for patch

        if (Uintah::Parallel::usingMPI()) {
          //Gather size from all processors
          int mycount = myPatchIDs.size();
          std::vector<int> counts(d_myworld->nRanks());
          Uintah::MPI::Allgather(&mycount, 1, MPI_INT, &counts[0], 1, MPI_INT, d_myworld->getComm());

          //compute recieve array offset and size
          std::vector<int> displs(d_myworld->nRanks());
          int pos = 0;

          for (int p = 0; p < d_myworld->nRanks(); p++) {
            displs[p] = pos;
            pos += counts[p];
          }

          std::vector<int> allPatchIDs(pos);  //receive array;
          Uintah::MPI::Allgatherv(&myPatchIDs[0], counts[d_myworld->myRank()], MPI_INT, &allPatchIDs[0], &counts[0], &displs[0], MPI_INT, d_myworld->getComm());

          //make refinePatchSets from patch ids
          std::set<int> allPatchIDset(allPatchIDs.begin(), allPatchIDs.end());

          for (UnstructuredLevel::patch_iterator iter = newLevel->patchesBegin(); iter != newLevel->patchesEnd(); ++iter) {
            UnstructuredPatch* newPatch = *iter;
            if (allPatchIDset.find(newPatch->getID()) != allPatchIDset.end()) {
              refinePatchSets[L]->add(newPatch);
            } else {
              copyPatchSets[L]->add(newPatch);
            }
          }
        }  // using MPI
      }

      if (refinePatchSets[L]->size() > 0) {
        DOUT(g_schedulercommon_dbg, "Rank-" << d_myworld->myRank() << "  Calling scheduleRefine for patches " << *refinePatchSets[L].get_rep());
        m_application->scheduleRefine(refinePatchSets[L].get_rep(), sched);
      }

    } else {
      refinePatchSets[L] = scinew UnstructuredPatchSet;
      copyPatchSets[L] = const_cast<UnstructuredPatchSet*>(newLevel->eachPatch());
    }

    //__________________________________
    //  Scheduling for copyDataToNewGrid
    if (copyPatchSets[L]->size() > 0) {
      dataTasks.push_back(scinew UnstructuredTask("UnstructuredSchedulerCommon::copyDataToNewGrid", this, &UnstructuredSchedulerCommon::copyDataToNewGrid));

      for (LabelMatlMap::iterator iter = m_label_matls[L].begin(); iter != m_label_matls[L].end(); iter++) {
        const UnstructuredVarLabel* var = iter->first;
        MaterialSubset* matls = iter->second;

        dataTasks.back()->requires(UnstructuredTask::OldDW, var, 0, UnstructuredTask::OtherGridDomain, matls, UnstructuredTask::NormalDomain, Ghost::None, 0);
        DOUT(g_schedulercommon_dbg, "  Scheduling copy for var " << *var << " matl " << *matls << " Copies: " << *copyPatchSets[L].get_rep());
        dataTasks.back()->computes(var, matls);
      }
      addTask(dataTasks.back(), copyPatchSets[L].get_rep(), m_sharedState->allMaterials());

      // Monitoring tasks must be scheduled last!!
      scheduleTaskMonitoring( copyPatchSets[L].get_rep() );      
    }

    //__________________________________
    //  Scheduling for modifyDataOnNewGrid
    if (refinePatchSets[L]->size() > 0) {
      dataTasks.push_back(scinew UnstructuredTask("UnstructuredSchedulerCommon::modifyDataOnNewGrid", this, &UnstructuredSchedulerCommon::copyDataToNewGrid));

      for (LabelMatlMap::iterator iter = m_label_matls[L].begin(); iter != m_label_matls[L].end(); iter++) {
        const UnstructuredVarLabel* var = iter->first;
        MaterialSubset* matls = iter->second;

        dataTasks.back()->requires(UnstructuredTask::OldDW, var, nullptr, UnstructuredTask::OtherGridDomain, matls, UnstructuredTask::NormalDomain, Ghost::None, 0);
        DOUT(g_schedulercommon_dbg, "  Scheduling modify for var " << *var << " matl " << *matls << " Modifies: " << *refinePatchSets[L].get_rep());
        dataTasks.back()->modifies(var, matls);
      }
      addTask(dataTasks.back(), refinePatchSets[L].get_rep(), m_sharedState->allMaterials());

      // Monitoring tasks must be scheduled last!!
      scheduleTaskMonitoring( refinePatchSets[L].get_rep());      
    }

    //__________________________________
    //  Component's shedule for refineInterfae
    if (L > 0) {
      m_application->scheduleRefineInterface(newLevel, sched, 0, 1);
    }
  }

  // set so the load balancer will make an adequate neighborhood, as
  // the default neighborhood isn't good enough for the copy data
  // timestep
  m_is_copy_data_timestep = true;  //-- do we still need this?  - BJW

#if !defined( DISABLE_SCI_MALLOC )
  const char* tag = AllocatorSetDefaultTag("DoDataCopy");
#endif

  this->compile();

  (*d_runtimeStats)[RegriddingCompilationTime] += timer().seconds();

  // save these and restore them, since the next execute will append the scheduler's, and we don't want to.
  double exec_time   = (*d_runtimeStats)[TaskExecTime];
  double local_time  = (*d_runtimeStats)[TaskLocalCommTime];
  double wait_time   = (*d_runtimeStats)[TaskWaitCommTime];
  double reduce_time = (*d_runtimeStats)[TaskReduceCommTime];
  double thread_time = (*d_runtimeStats)[TaskWaitThreadTime];

  timer.reset( true );
  this->execute();

#if !defined( DISABLE_SCI_MALLOC )
  AllocatorSetDefaultTag(tag);
#endif

  //__________________________________
  // copy reduction variables to the new_dw
  std::vector<UnstructuredVarLabelMatl<UnstructuredLevel> > levelVariableInfo;
  oldDataWarehouse->getVarLabelMatlLevelTriples(levelVariableInfo);

  newDataWarehouse->unfinalize();
  for (unsigned int i = 0; i < levelVariableInfo.size(); i++) {
    UnstructuredVarLabelMatl<UnstructuredLevel> currentReductionVar = levelVariableInfo[i];

    if (currentReductionVar.label_->typeDescription()->isReductionVariable()) {

      // cout << "REDUNCTION:  Label(" << setw(15) << currentReductionVar.label_->getName() << "): Patch(" << reinterpret_cast<int>(currentReductionVar.level_) << "): Material(" << currentReductionVar.matlIndex_ << ")" << endl; 
      const UnstructuredLevel* oldLevel = currentReductionVar.domain_;
      const UnstructuredLevel* newLevel = nullptr;
      if (oldLevel && oldLevel->getIndex() < grid->numLevels()) {

        if (oldLevel->getIndex() >= grid->numLevels()) {
          // the new grid no longer has this level
          continue;
        }
        newLevel = (newDataWarehouse->getGrid()->getLevel(oldLevel->getIndex())).get_rep();
      }

      //  Either both levels need to be null or both need to exist (null levels mean global data)
      if (!oldLevel || newLevel) {
        UnstructuredReductionVariableBase* v = dynamic_cast<UnstructuredReductionVariableBase*>(currentReductionVar.label_->typeDescription()->createInstance());

        oldDataWarehouse->get(*v, currentReductionVar.label_, currentReductionVar.domain_, currentReductionVar.matlIndex_);
        newDataWarehouse->put(*v, currentReductionVar.label_, newLevel, currentReductionVar.matlIndex_);
        delete v;  // copied on the put command
      }
    }
  }

  newDataWarehouse->refinalize();

  (*d_runtimeStats)[RegriddingCopyDataTime] += timer().seconds();

  // restore values from before the regrid and data copy
  (*d_runtimeStats)[TaskExecTime]       = exec_time;
  (*d_runtimeStats)[TaskLocalCommTime]  = local_time;
  (*d_runtimeStats)[TaskWaitCommTime]   = wait_time;
  (*d_runtimeStats)[TaskReduceCommTime] = reduce_time;
  (*d_runtimeStats)[TaskWaitThreadTime] = thread_time;

  m_is_copy_data_timestep = false;
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::copyDataToNewGrid( const ProcessorGroup * /* pg */
                                  , const UnstructuredPatchSubset    * patches
                                  , const MaterialSubset * matls
                                  ,       UnstructuredDataWarehouse  * old_dw
                                  ,       UnstructuredDataWarehouse  * new_dw
                                  )
{
  DOUT(g_schedulercommon_dbg, "UnstructuredSchedulerCommon::copyDataToNewGrid() BGN on patches " << *patches);

  UnstructuredOnDemandDataWarehouse* oldDataWarehouse = dynamic_cast<UnstructuredOnDemandDataWarehouse*>(old_dw);
  UnstructuredOnDemandDataWarehouse* newDataWarehouse = dynamic_cast<UnstructuredOnDemandDataWarehouse*>(new_dw);

  // For each patch in the patch subset which contains patches in the new grid
  for (int p = 0; p < patches->size(); p++) {
    const UnstructuredPatch* newPatch = patches->get(p);
    const UnstructuredLevel* newLevel = newPatch->getLevel();

    // to create once per matl instead of once per matl-var
    std::vector<UnstructuredParticleSubset*> oldsubsets(m_sharedState->getNumMatls()), newsubsets(m_sharedState->getNumMatls());

    // If there is a level that didn't exist, we don't need to copy it
    if (newLevel->getIndex() >= oldDataWarehouse->getGrid()->numLevels()) {
      continue;
    }

    // find old patches associated with this patch
    UnstructuredLevelP oldLevel = oldDataWarehouse->getGrid()->getLevel(newLevel->getIndex());

    //__________________________________
    //  Loop over Var labels
    for (LabelMatlMap::iterator iter = m_label_matls[oldLevel->getIndex()].begin();
         iter != m_label_matls[oldLevel->getIndex()].end(); iter++) {
      const UnstructuredVarLabel* label = iter->first;
      MaterialSubset* var_matls = iter->second;

      // get the low/high for what we'll need to get
      UnstructuredPatch::VariableBasis basis = UnstructuredPatch::translateTypeToBasis(label->typeDescription()->getUnstructuredType(), true);
      IntVector newLowIndex, newHighIndex;
      newPatch->computeVariableExtents(basis, IntVector(0, 0, 0), Ghost::None, 0, newLowIndex, newHighIndex);

      //__________________________________
      //  Loop over materials
      for (int m = 0; m < var_matls->size(); m++) {
        int matl = var_matls->get(m);

        if (!matls->contains(matl)) {
          continue;
        }

        //__________________________________
        //  Grid Variables
        if (label->typeDescription()->getUnstructuredType() != UnstructuredTypeDescription::UnstructuredParticleVariable) {
          UnstructuredPatch::selectType oldPatches;
          oldLevel->selectPatches(newLowIndex, newHighIndex, oldPatches);

          for (unsigned int oldIdx = 0; oldIdx < oldPatches.size(); oldIdx++) {
            const UnstructuredPatch* oldPatch = oldPatches[oldIdx];

            if (!oldDataWarehouse->exists(label, matl, oldPatch)) {
              continue;  // see comment about oldPatchToTest in ScheduleAndDoDataCopy
            }

            IntVector oldLowIndex;
            IntVector oldHighIndex;

            if (newLevel->getIndex() > 0) {
              oldLowIndex = oldPatch->getLowIndexWithDomainLayer(basis);
              oldHighIndex = oldPatch->getHighIndexWithDomainLayer(basis);
            } else {
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

            switch (label->typeDescription()->getUnstructuredType()) {
              case UnstructuredTypeDescription::NCVariable :
              case UnstructuredTypeDescription::CCVariable :
              case UnstructuredTypeDescription::SFCXVariable :
              case UnstructuredTypeDescription::SFCYVariable :
              case UnstructuredTypeDescription::SFCZVariable : {
                // bulletproofing
                if (!oldDataWarehouse->exists(label, matl, oldPatch)) {
                  SCI_THROW(
                      UnknownVariable(label->getName(), oldDataWarehouse->getID(), oldPatch, matl, "in copyDataTo GridVariableBase", __FILE__, __LINE__));
                }

                std::vector<UnstructuredVariable *> varlist;
                oldDataWarehouse->d_varDB.getlist(label, matl, oldPatch, varlist);
                UnstructuredGridVariableBase* v = nullptr;

                IntVector srclow = copyLowIndex;
                IntVector srchigh = copyHighIndex;

                for (unsigned int i = 0; i < varlist.size(); ++i) {
                  v = dynamic_cast<UnstructuredGridVariableBase*>(varlist[i]);

                  ASSERT(v->getBasePointer() != nullptr);

                  //restrict copy to data range
                  srclow = Max(copyLowIndex, v->getLow());
                  srchigh = Min(copyHighIndex, v->getHigh());

                  if (srclow.x() >= srchigh.x() || srclow.y() >= srchigh.y() || srclow.z() >= srchigh.z()) {
                    continue;
                  }

                  if (!newDataWarehouse->exists(label, matl, newPatch)) {

                    UnstructuredGridVariableBase* newVariable = v->cloneType();
                    newVariable->rewindow(newLowIndex, newHighIndex);
                    newVariable->copyPatch(v, srclow, srchigh);
                    newDataWarehouse->d_varDB.put(label, matl, newPatch, newVariable, copyTimestep(), false);

                  } else {
                    UnstructuredGridVariableBase* newVariable = dynamic_cast<UnstructuredGridVariableBase*>(newDataWarehouse->d_varDB.get(label, matl, newPatch));
                    // make sure it exists in the right region (it might be ghost data)
                    newVariable->rewindow(newLowIndex, newHighIndex);

                    if (oldPatch->isVirtual()) {
                      // it can happen where the old patch was virtual and this is not
                      UnstructuredGridVariableBase* tmpVar = newVariable->cloneType();
                      tmpVar->copyPointer(*v);
                      tmpVar->offset(oldPatch->getVirtualOffset());
                      newVariable->copyPatch(tmpVar, srclow, srchigh);
                      delete tmpVar;
                    } else {
                      newVariable->copyPatch(v, srclow, srchigh);

                    }
                  }
                }
              }
                break;
              case UnstructuredTypeDescription::UnstructuredPerPatch : {
              }
                break;
              default :
                SCI_THROW(InternalError("Unknown variable type in copyData: "+label->getName(), __FILE__, __LINE__));
            }  // end switch
          }  // end oldPatches
        } else {
          //__________________________________
          //  Particle Variables
          UnstructuredParticleSubset* oldsub = oldsubsets[matl];
          if (!oldsub) {
            // collect the particles from the range encompassing this patch.  Use interior cells since
            // extracells aren't collected across processors in the data copy, and they don't matter
            // for particles anyhow (but we will have to reset the bounds to copy the data)
            oldsub = oldDataWarehouse->getParticleSubset(matl, newPatch->getLowIndexWithDomainLayer(UnstructuredPatch::CellBased),
                                                         newPatch->getHighIndexWithDomainLayer(UnstructuredPatch::CellBased), newPatch,
                                                         m_reloc_new_pos_label, oldLevel.get_rep());
            oldsubsets[matl] = oldsub;
            oldsub->addReference();
          }

          UnstructuredParticleSubset* newsub = newsubsets[matl];
          // it might have been created in Refine
          if (!newsub) {
            if (!newDataWarehouse->haveParticleSubset(matl, newPatch)) {
              newsub = newDataWarehouse->createParticleSubset(oldsub->numParticles(), matl, newPatch);
            } else {
              newsub = newDataWarehouse->getParticleSubset(matl, newPatch);
              ASSERT(newsub->numParticles() == 0);
              newsub->addParticles(oldsub->numParticles());
            }
            newsubsets[matl] = newsub;
          }

          UnstructuredParticleVariableBase* newv = dynamic_cast<UnstructuredParticleVariableBase*>(label->typeDescription()->createInstance());
          newv->allocate(newsub);
          // don't get and copy if there were no old patches
          if (oldsub->getNeighbors().size() > 0) {

            constUnstructuredParticleVariableBase* var = newv->cloneConstType();
            oldDataWarehouse->get(*var, label, oldsub);

            // reset the bounds of the old var's data so copyData doesn't complain
            UnstructuredParticleSubset* tempset = scinew UnstructuredParticleSubset(oldsub->numParticles(), matl, newPatch,
                                                            newPatch->getExtraCellLowIndex(), newPatch->getExtraCellHighIndex());
            const_cast<UnstructuredParticleVariableBase*>(&var->getBaseRep())->setParticleSubset(tempset);
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

  DOUT(g_schedulercommon_dbg, "UnstructuredSchedulerCommon::copyDataToNewGrid() END");
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::scheduleParticleRelocation( const UnstructuredLevelP       & level
                                           , const UnstructuredVarLabel     * old_posLabel
                                           , const VarLabelList & old_labels
                                           , const UnstructuredVarLabel     * new_posLabel
                                           , const VarLabelList & new_labels
                                           , const UnstructuredVarLabel     * particleIDLabel
                                           , const MaterialSet  * matls
                                           ,       int            which
                                           )
{
  if (which == 1) {
    if (m_reloc_new_pos_label) {
      ASSERTEQ(m_reloc_new_pos_label, new_posLabel);
    }
    m_reloc_new_pos_label = new_posLabel;

    m_relocate_1.scheduleParticleRelocation(this, d_myworld, m_loadBalancer, level, old_posLabel, old_labels, new_posLabel, new_labels, particleIDLabel, matls);
    releasePort("load balancer");
  }

  if (which == 2) {
    if (m_reloc_new_pos_label) {
      ASSERTEQ(m_reloc_new_pos_label, new_posLabel);
    }
    m_reloc_new_pos_label = new_posLabel;

    m_relocate_2.scheduleParticleRelocation(this, d_myworld, m_loadBalancer, level, old_posLabel, old_labels, new_posLabel, new_labels, particleIDLabel, matls);
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::scheduleParticleRelocation( const UnstructuredLevelP       & coarsestLevelwithParticles
                                           , const UnstructuredVarLabel     * old_posLabel
                                           , const VarLabelList & old_labels
                                           , const UnstructuredVarLabel     * new_posLabel
                                           , const VarLabelList & new_labels
                                           , const UnstructuredVarLabel     * particleIDLabel
                                           , const MaterialSet  * matls
                                           )
{
  if (m_reloc_new_pos_label) {
    ASSERTEQ(m_reloc_new_pos_label, new_posLabel);
  }

  m_reloc_new_pos_label = new_posLabel;

  m_relocate_1.scheduleParticleRelocation(this, d_myworld, m_loadBalancer, coarsestLevelwithParticles, old_posLabel, old_labels, new_posLabel, new_labels, particleIDLabel, matls);
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::scheduleParticleRelocation( const UnstructuredLevelP       & coarsestLevelwithParticles
                                           , const UnstructuredVarLabel     * posLabel
                                           , const VarLabelList & otherLabels
                                           , const MaterialSet  * matls
                                           )
{
  m_reloc_new_pos_label = posLabel;

  m_relocate_1.scheduleParticleRelocation(this, d_myworld, m_loadBalancer, coarsestLevelwithParticles, posLabel, otherLabels, matls);
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::overrideVariableBehavior( const std::string & var
                                         ,       bool          treatAsOld
                                         ,       bool          copyData
                                         ,       bool          noScrub
                                         ,       bool          notCopyData
                                         ,       bool          noCheckpoint
                                         )
{
  // treat variable as an "old" var - will be checkpointed, copied, and only scrubbed from an OldDW
  if (treatAsOld) {
    m_treat_as_old_vars.insert(var);
  }

  // manually copy variable between AMR levels
  if (copyData) {
    m_copy_data_vars.insert(var);
    m_no_scrub_vars.insert(var);
  }

  // set variable not to scrub (normally when needed between a normal taskgraph
  // and the regridding phase)
  if (noScrub) {
    m_no_scrub_vars.insert(var);
  }

  // ignore copying this variable between AMR levels
  if (notCopyData) {
    m_no_copy_data_vars.insert(var);
  }

  // do not checkpoint this variable.
  if (noCheckpoint) {
    m_no_checkpoint_vars.insert(var);
  }
}

//______________________________________________________________________
//
void
UnstructuredSchedulerCommon::clearTaskMonitoring()
{
  // Loop through the global (0) and local (1) tasks
  for (unsigned int i = 0; i < 2; ++i)
    m_monitoring_values[i].clear();
}

//______________________________________________________________________
// Schedule the recording of the task monitoring attribute
// values. This task should be the last task so that the writing is
// done after all task have been executed.
void
UnstructuredSchedulerCommon::scheduleTaskMonitoring( const UnstructuredLevelP& level )
{
  if( !m_monitoring )
    return;

  // Create and schedule a task that will record each of the
  // tasking monitoring attributes.
  UnstructuredTask* t = scinew UnstructuredTask("UnstructuredSchedulerCommon::recordTaskMonitoring",
                        this, &UnstructuredSchedulerCommon::recordTaskMonitoring);

  // Ghost::GhostType gn = Ghost::None;

  for (unsigned int i = 0; i < 2; ++i)
  {
    for( const auto &it : m_monitoring_tasks[i] )
    {
      t->computes( it.second, m_dummy_matl, UnstructuredTask::OutOfDomain );
      
      overrideVariableBehavior(it.second->getName(),
                               false, false, true, true, true);
      // treatAsOld copyData noScrub notCopyData noCheckpoint
    }
  }

  addTask(t, level->eachPatch(), m_sharedState->allMaterials());
}

//______________________________________________________________________
// Schedule the recording of the task monitoring attribute
// values. This task should be the last task so that the writing is
// done after all task have been executed.
void
UnstructuredSchedulerCommon::scheduleTaskMonitoring( const UnstructuredPatchSet* patches )
{
  if( !m_monitoring )
    return;

  // Create and schedule a task that will record each of the
  // tasking monitoring attributes.
  UnstructuredTask* t = scinew UnstructuredTask("UnstructuredSchedulerCommon::recordTaskMonitoring",
                        this, &UnstructuredSchedulerCommon::recordTaskMonitoring);

  // Ghost::GhostType gn = Ghost::None;

  for (unsigned int i = 0; i < 2; ++i)
  {
    for( const auto &it : m_monitoring_tasks[i] )
    {
      t->computes( it.second, m_dummy_matl, UnstructuredTask::OutOfDomain );

      overrideVariableBehavior(it.second->getName(),
                               false, false, true, true, true);
      // treatAsOld copyData noScrub notCopyData noCheckpoint
    }
  }

  addTask(t, patches, m_sharedState->allMaterials());
}

//______________________________________________________________________
// Record the global task monitoring attribute values into the data
// warehouse.
void UnstructuredSchedulerCommon::recordTaskMonitoring( const ProcessorGroup *
                                          , const UnstructuredPatchSubset    * patches
                                          , const MaterialSubset * /*matls*/
                                          ,       UnstructuredDataWarehouse  * old_dw
                                          ,       UnstructuredDataWarehouse  * new_dw
                                          )
{
  int matlIndex = 0;

  // For all of the patches record the tasking monitoring attribute value.
  for (int p = 0; p < patches->size(); p++) {
    const UnstructuredPatch* patch = patches->get(p);

    // Loop through the global (0) and local (1) tasks
    for (unsigned int i = 0; i < 2; ++i) {
      for (const auto &it : m_monitoring_tasks[i]) {
        UnstructuredPerPatch<double> value = m_monitoring_values[i][it.first][patch->getID()];

        new_dw->put(value, it.second, matlIndex, patch);
      }
    }
  }
}

//______________________________________________________________________
// Sum the task monitoring attribute values
void
UnstructuredSchedulerCommon::sumTaskMonitoringValues( UnstructuredDetailedTask * dtask )
{
  if (!m_monitoring) {
    return;
  }

  const UnstructuredPatchSubset *patches = dtask->getPatches();

  if (patches && patches->size()) {
    // Compute the cost on a per cell basis so the measured value can
    // be distributed proportionally by cells
    double num_cells = 0;

    for (int p = 0; p < patches->size(); p++) {
      const UnstructuredPatch* patch = patches->get(p);
      num_cells += patch->getNumExtraCells();
    }

    double weight;

    // Compute the value on a per cell basis.
    if (m_monitoring_per_cell) {
      weight = num_cells;
    }
    // Compute the value on a per patch basis.
    else {
      weight = 1;
    }

    // Loop through the global (0) and local (1) tasks
    for (auto i = 0; i < 2; ++i) {
      for (const auto &it : m_monitoring_tasks[i]) {
        // Strip off the attribute name from the task name.
        std::string taskName = it.first;

        // For a local task strip off the attribute name.
        if (i == 1) {
          size_t found = taskName.find_last_of("::");
          // std::string attribute = taskName.substr(found + 1);
          taskName = taskName.substr(0, found - 1);
        }

        // Is this task being monitored ?
        if ((i == 0) ||  // Global monitoring yes, otherwise check.
            (i == 1 && taskName == dtask->getTask()->getName())) {
          bool loadBalancerCost = false;
          double value;

          // Currently the monitoring is limited to the LoadBalancer
          // cost, task exec time, and task wait time.
          if (it.first.find("LoadBalancerCost") != std::string::npos) {
            // The same code is in runTask of the specific scheduler
            // (MPIScheduler and UnifiedScheduler) to use the task
            // execution time which is then weighted by the number of
            // cells in CostModelForecaster::addContribution
            if (!dtask->getTask()->getHasSubScheduler() && !m_is_copy_data_timestep && dtask->getTask()->getType() != UnstructuredTask::Output) {
              value = dtask->task_exec_time() / num_cells;
              loadBalancerCost = true;
            }
            else {
              value = 0.0;
            }
          }
          else if (it.first.find("ExecTime") != std::string::npos) {
            value = dtask->task_exec_time() / weight;
          }
          else if (it.first.find("WaitTime") != std::string::npos) {
            value = dtask->task_wait_time() / weight;
          }
          else {
            value = 0.0;
          }

          if (value != 0.0) {

            // Loop through patches and add the contribution.
            for (int p = 0; p < patches->size(); ++p) {
              const UnstructuredPatch* patch = patches->get(p);

              if (m_monitoring_per_cell || loadBalancerCost) {
                m_monitoring_values[i][it.first][patch->getID()] += patch->getNumExtraCells() * value;
              }
              else {
                m_monitoring_values[i][it.first][patch->getID()] += value;
              }

              // A cheat ... the only time this task will come here is
              // after the value has been written (and the task is
              // completed) so the value can be overwritten. This
              // allows the monitoring to be monitored.
              if (dtask->getTask()->getName() == "UnstructuredSchedulerCommon::recordTaskMonitoring") {
                UnstructuredPerPatch<double> value = m_monitoring_values[i][it.first][patch->getID()];
                this->getLastDW()->put(value, it.second, 0, patch);
              }
            }
          }
        }
      }
    }
  }
}
