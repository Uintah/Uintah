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

#include <CCA/Components/DataArchiver/DataArchiver.h>

#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/OutputContext.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/ApplicationInterface.h>

#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/Endian.h>
#include <Core/Util/Environment.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/FileUtils.h>
#include <Core/Util/StringUtil.h>
#include <Core/Util/Timers/Timers.hpp>

#include <sci_defs/visit_defs.h>

#include <iomanip>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <sstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cmath>
#include <cstring>

#include <sys/param.h>
#include <strings.h>
#include <unistd.h>

#include <libxml/xmlwriter.h>


//TODO - BJW - if multilevel reduction doesn't work, fix all
//       getMaterialSet(0)

#define PADSIZE    1024L
#define ALL_LEVELS   99

#define OUTPUT               0
#define CHECKPOINT           1
#define CHECKPOINT_REDUCTION 2

#define XML_TEXTWRITER 1
#undef XML_TEXTWRITER

using namespace Uintah;
using namespace std;

static DebugStream dbg("DataArchiver", false);

#ifdef HAVE_PIDX
  static DebugStream dbgPIDX ("DataArchiverPIDX", false);
#endif

bool DataArchiver::m_wereSavesAndCheckpointsInitialized = false;

DataArchiver::DataArchiver(const ProcessorGroup* myworld, int udaSuffix)
  : UintahParallelComponent(myworld),
    m_udaSuffix(udaSuffix)
{
  m_isOutputTimeStep      = false;
  m_isCheckpointTimeStep  = false;
  m_saveParticleVariables = false;
  m_saveP_x               = false;
  m_particlePositionName  = "p.x";
  m_doPostProcessUda      = false;
  m_outputFileFormat      = UDA;

  m_XMLIndexDoc = nullptr;
  m_CheckpointXMLIndexDoc = nullptr;

  m_outputDoubleAsFloat = false;

  m_fileSystemRetrys = 10;
  m_numLevelsInOutput = 0;

  m_writeMeta = false;

  // Time Step
  m_timeStepLabel =
    VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );
  // Simulation Time
  m_simTimeLabel =
    VarLabel::create(simTime_name, simTime_vartype::getTypeDescription() );
  // Simulation Time
  m_delTLabel =
    VarLabel::create(delT_name, delt_vartype::getTypeDescription() );
}

DataArchiver::~DataArchiver()
{
  VarLabel::destroy(m_timeStepLabel);
  VarLabel::destroy(m_simTimeLabel);
  VarLabel::destroy(m_delTLabel);
}

//______________________________________________________________________
//
void
DataArchiver::getComponents()
{
  m_application = dynamic_cast<ApplicationInterface*>( getPort("application") );

  if( !m_application ) {
    throw InternalError("dynamic_cast of 'm_application' failed!", __FILE__, __LINE__);
  }

  m_loadBalancer = dynamic_cast<LoadBalancer*>( getPort("load balancer") );

  if( !m_loadBalancer ) {
    throw InternalError("dynamic_cast of 'm_loadBalancer' failed!", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
void
DataArchiver::releaseComponents()
{
  releasePort( "application" );
  releasePort( "load balancer" );

  m_application  = nullptr;
  m_loadBalancer = nullptr;

  m_sharedState  = nullptr;
}

//______________________________________________________________________
//
void
DataArchiver::problemSetup( const ProblemSpecP    & params,
			    const ProblemSpecP    & restart_prob_spec,
			    const SimulationStateP& sharedState )
{
  if (dbg.active()) {
    dbg << "Doing ProblemSetup \t\t\t\tDataArchiver\n";
  }

  m_sharedState = sharedState;
  m_upsFile = params;
  ProblemSpecP p = params->findBlock("DataArchiver");
  
  if( restart_prob_spec ) {

    ProblemSpecP insitu_ps = restart_prob_spec->findBlock( "InSitu" );

    if( insitu_ps != nullptr ) {

      bool haveModifiedVars;
      
      insitu_ps = insitu_ps->get("haveModifiedVars", haveModifiedVars);

      m_application->haveModifiedVars( haveModifiedVars );

      if (haveModifiedVars) {
	      std::stringstream tmpstr;
	      tmpstr << "DataArchiver found previously modified variables that "
	             << "have not been merged into the checkpoint restart "
	             << "input.xml file from the from index.xml file. " << std::endl
	             << "The modified variables can be found in the "
	             << "index.xml file under the 'InSitu' block." << std::endl
	             << "Once merged, change the variable 'haveModifiedVars' in "
	             << "the 'InSitu' block in the checkpoint restart timestep.xml "
	             << "file to 'false'";
	
	      throw ProblemSetupException(tmpstr.str(), __FILE__, __LINE__);
      }
      else {
        proc0cout << "DataArchiver found previously modified vars. "
                  << "Assuming the checkpoint restart input.xml file "
                  << "has been updated." << std::endl;
      }
    }
  }
  
  //__________________________________
  // PIDX related
  string type;
  p->getAttribute("type", type);
  if (type == "pidx" || type == "PIDX") {
    m_outputFileFormat = PIDX;
    m_PIDX_flags.problemSetup(p);
    //m_PIDX_flags.print();
  }

  m_outputDoubleAsFloat = p->findBlock("outputDoubleAsFloat") != nullptr;

  // set to false if restartSetup is called - we can't do it there
  // as the first timestep doesn't have any tasks
  m_outputInitTimeStep = p->findBlock("outputInitTimestep") != nullptr;
  
  // problemSetup is called again from the Switcher to reset vars (and
  // frequency) it wants to save DO NOT get it again.  Currently the
  // directory won't change mid-run, so calling problemSetup will not
  // change the directory.  What happens then, is even if a switched
  // component wants a different uda name, it will not get one until
  // sus restarts (i.e., when you switch, component 2's data dumps
  // will be in whichever uda started sus.), which is not optimal.  So
  // we disable this feature until we can make the DataArchiver make a
  // new directory mid-run.
  if (m_filebase == "") {
    p->require("filebase", m_filebase);
  }

  // Get output timestep interval, or time interval info:
  m_outputInterval = 0;
  if( !p->get( "outputTimestepInterval", m_outputTimeStepInterval ) ) {
    m_outputTimeStepInterval = 0;
  }
  
  if ( !p->get("outputInterval", m_outputInterval) && m_outputTimeStepInterval == 0 ) {
    m_outputInterval = 0.0; // default
  }

  if ( m_outputInterval > 0.0 && m_outputTimeStepInterval > 0 ) {
    throw ProblemSetupException("Use <outputInterval> or <outputTimeStepInterval>, not both",__FILE__, __LINE__);
  }

  if ( !p->get("outputLastTimestep", m_outputLastTimeStep) ) {
    m_outputLastTimeStep = false; // default
  }

  // set default compression mode - can be "tryall", "gzip", "rle",
  // "rle, gzip", "gzip, rle", or "none"
  string defaultCompressionMode = "";
  if (p->get("compression", defaultCompressionMode)) {
    VarLabel::setDefaultCompressionMode(defaultCompressionMode);
  }

  if (params->findBlock("ParticlePosition")) {
    params->findBlock("ParticlePosition")->getAttribute("label",m_particlePositionName);
  }

  //__________________________________
  // parse the variables to be saved
  m_saveLabelNames.clear(); // we can problemSetup multiple times on a component Switch, clear the old ones.
  map<string, string> attributes;
  SaveNameItem saveItem;
  ProblemSpecP save = p->findBlock("save");

  if( save == nullptr ) {
    // If no <save> labels were specified, make sure that an output
    // time interval is not specified...
    if( m_outputInterval > 0.0 || m_outputTimeStepInterval > 0 ) {
      throw ProblemSetupException( "You have no <save> labels, but your output interval is non-0.  If you wish to turn off "
                                   "data output, you must set <outputTimestepInterval> or <outputInterval> to 0.",
                                   __FILE__, __LINE__);
    }
  }

  while( save != nullptr ) {
    attributes.clear();
    save->getAttributes(attributes);
    saveItem.labelName       = attributes["label"];
    saveItem.compressionMode = attributes["compression"];
    
    try {
      saveItem.matls = ConsecutiveRangeSet(attributes["material"]);
    }
    catch (ConsecutiveRangeSetException&) {
      throw ProblemSetupException("'" + attributes["material"] + "'" +
             " cannot be parsed as a set of material" +
             " indices for saving '" + saveItem.labelName + "'",
                                  __FILE__, __LINE__);
    }
    
    // if materials aren't specified, all valid materials will be saved
    if (saveItem.matls.size() == 0){  
      saveItem.matls = ConsecutiveRangeSet::all;
    }


    try {
      // level values are normal for exact level values (i.e.,
      //   0, 2-4 will save those levels.
      //   Leave blank for all levels
      //   Also handle negative values to be relative to the finest levels
      //   I.e., -3--1 would be the top three levels.
      saveItem.levels = ConsecutiveRangeSet(attributes["levels"]);
    }
    catch (ConsecutiveRangeSetException&) {
      throw ProblemSetupException("'" + attributes["levels"] + "'" +
             " cannot be parsed as a set of levels" +
             " for saving '" + saveItem.labelName + "'",
                                  __FILE__, __LINE__);
    }
    
    // if levels aren't specified, all valid materials will be saved
    if (saveItem.levels.size() == 0) {
      saveItem.levels = ConsecutiveRangeSet(ALL_LEVELS, ALL_LEVELS);
    }
    
    //__________________________________
    //  bullet proofing: must save p.x 
    //  in addition to other particle variables "p.*"
    if (saveItem.labelName == m_particlePositionName || saveItem.labelName == "p.xx") {
      m_saveP_x = true;
    }

    string::size_type pos = saveItem.labelName.find("p.");
    if ( pos != string::npos &&  saveItem.labelName != m_particlePositionName) {
      m_saveParticleVariables = true;
    }

    m_saveLabelNames.push_back( saveItem );

    save = save->findNextBlock( "save" );
  }
  
  if(m_saveP_x == false && m_saveParticleVariables == true) {
    throw ProblemSetupException(" You must save " + m_particlePositionName + " when saving other particle variables", __FILE__, __LINE__);
  }     

  //__________________________________
  // get checkpoint information
  m_checkpointInterval         = 0.0;
  m_checkpointTimeStepInterval = 0;
  m_checkpointWallTimeStart    = 0;
  m_checkpointWallTimeInterval = 0;
  m_checkpointCycle            = 2; /* 2 is the smallest number that is safe
                                    (always keeping an older copy for backup) */
  m_checkpointLastTimeStep     = false;
  
  ProblemSpecP checkpoint = p->findBlock( "checkpoint" );
  if( checkpoint != nullptr ) {

    string interval, timestepInterval, wallTimeStart, wallTimeInterval,
      wallTimeStartHours, wallTimeIntervalHours, cycle, lastTimeStep;

    attributes.clear();
    checkpoint->getAttributes( attributes );

    interval              = attributes[ "interval" ];
    timestepInterval      = attributes[ "timestepInterval" ];
    wallTimeStart         = attributes[ "walltimeStart" ];
    wallTimeInterval      = attributes[ "walltimeInterval" ];
    wallTimeStartHours    = attributes[ "walltimeStartHours" ];
    wallTimeIntervalHours = attributes[ "walltimeIntervalHours" ];
    cycle                 = attributes[ "cycle" ];
    lastTimeStep          = attributes[ "lastTimestep" ];

    if( interval != "" ) {
      m_checkpointInterval = atof( interval.c_str() );
    }
    if( timestepInterval != "" ) {
      m_checkpointTimeStepInterval = atoi( timestepInterval.c_str() );
    }
    if( wallTimeStart != "" ) {
      m_checkpointWallTimeStart = atoi( wallTimeStart.c_str() );
    }      
    if( wallTimeInterval != "" ) {
      m_checkpointWallTimeInterval = atoi( wallTimeInterval.c_str() );
    }
    if( wallTimeStartHours != "" ) {
      m_checkpointWallTimeStart = atof( wallTimeStartHours.c_str() ) * 3600.0;
    }      
    if( wallTimeIntervalHours != "" ) {
      m_checkpointWallTimeInterval = atof( wallTimeIntervalHours.c_str() ) * 3600.0;
    }
    if( cycle != "" ) {
      m_checkpointCycle = atoi( cycle.c_str() );
    }
    if( lastTimeStep == "true" ) {
      m_checkpointLastTimeStep = true;
    }

    // Verify that an interval was specified:
    if( interval == "" && timestepInterval == "" &&
	wallTimeInterval == "" && wallTimeIntervalHours == "" ) {
      throw ProblemSetupException( "ERROR: \n  <checkpoint> must specify either interval, timestepInterval, walltimeInterval",
                                   __FILE__, __LINE__ );
    }
  }

  // Can't use both checkpointInterval and checkpointTimeStepInterval.
  if (m_checkpointInterval > 0.0 && m_checkpointTimeStepInterval > 0) {
    throw ProblemSetupException("Use <checkpoint interval=...> or <checkpoint timestepInterval=...>, not both",
                                __FILE__, __LINE__);
  }
  // Can't have a WallTimeStart without a WallTimeInterval.
  if (m_checkpointWallTimeStart > 0.0 && m_checkpointWallTimeInterval == 0) {
    throw ProblemSetupException("<checkpoint walltimeStart must have a corresponding walltimeInterval",
                                __FILE__, __LINE__);
  }

  m_lastTimeStepLocation   = "invalid";
  m_isOutputTimeStep       = false;

  // Set up the next output and checkpoint time. Always output the
  // first timestep or the inital timestep.
  m_nextOutputTime     = 0;
  m_nextOutputTimeStep = m_outputInitTimeStep ? 0 : 1;

  m_nextCheckpointTime     = m_checkpointInterval;
  m_nextCheckpointTimeStep = m_checkpointTimeStepInterval + 1;
  m_nextCheckpointWallTime = m_checkpointWallTimeStart;

  //__________________________________
  // 
  if ( m_checkpointInterval > 0 ) {
    proc0cout << "Checkpointing:" << std::setw(16) << " Every "
	      << m_checkpointInterval << " physical seconds.\n";
  }
  if ( m_checkpointTimeStepInterval > 0 ) {
    proc0cout << "Checkpointing:" << std::setw(16)<< " Every "
	      << m_checkpointTimeStepInterval << " timesteps.\n";
  }
  if ( m_checkpointWallTimeInterval > 0 ) {
    proc0cout << "Checkpointing:" << std::setw(16)<< " Every "
	      << m_checkpointWallTimeInterval << " wall clock seconds,"
              << " starting after " << m_checkpointWallTimeStart << " seconds.\n";
  }
  
#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( m_application->getVisIt() && !initialized ) {
    m_application->getDebugStreams().push_back( &dbg  );
#ifdef HAVE_PIDX
    m_application->getDebugStreams().push_back( &dbgPIDX );
#endif
    initialized = true;
  }
#endif
}

//______________________________________________________________________
//
void
DataArchiver::outputProblemSpec( ProblemSpecP & root_ps )
{
  if (dbg.active()) {
    dbg << "Doing outputProblemSpec \t\t\t\tDataArchiver\n";
  }

  if( m_application->haveModifiedVars() ) {

    ProblemSpecP root = root_ps->getRootNode();

    ProblemSpecP is_ps = root->findBlockWithOutAttribute( "InSitu" );

    if( is_ps == nullptr )
      is_ps = root->appendChild("InSitu");

    is_ps->appendElement("haveModifiedVars", m_application->haveModifiedVars());
  }
}


//______________________________________________________________________
//
void
DataArchiver::initializeOutput( const ProblemSpecP & params )
{
  if( m_outputInterval             == 0.0 && 
      m_outputTimeStepInterval     == 0   && 
      m_checkpointInterval         == 0.0 && 
      m_checkpointTimeStepInterval == 0   && 
      m_checkpointWallTimeInterval == 0 ) {
    return;
  }

  if( getUseLocalFileSystems() ) {
    setupLocalFileSystems();
  }
  else {
    setupSharedFileSystem();
  }
  // Wait for all ranks to finish verifying shared file system....
  Uintah::MPI::Barrier(d_myworld->getComm());

  if (m_writeMeta) {

    saveSVNinfo();
    // Create index.xml:
    string inputname = m_dir.getName()+"/input.xml";
    params->output( inputname.c_str() );

    /////////////////////////////////////////////////////////
    // Save the original .ups file in the UDA...
    //     FIXME: might want to avoid using 'system' copy which the
    //     below uses...  If so, we will need to write our own
    //     (simple) file reader and writer routine.

    cout << "Saving original .ups file in UDA...\n";
    Dir ups_location( pathname( params->getFile() ) );
    ups_location.copy( basename( params->getFile() ), m_dir );

    //
    /////////////////////////////////////////////////////////

    createIndexXML(m_dir);
   
    // create checkpoints/index.xml (if we are saving checkpoints)
    if ( m_checkpointInterval         > 0.0 || 
         m_checkpointTimeStepInterval > 0   || 
         m_checkpointWallTimeInterval > 0 ) {
      m_checkpointsDir = m_dir.createSubdir("checkpoints");
      createIndexXML(m_checkpointsDir);
    }
  }
  else {
    m_checkpointsDir = m_dir.getSubdir("checkpoints");
  }

  // Sync up before every rank can use the base dir.
  Uintah::MPI::Barrier(d_myworld->getComm());
} // end initializeOutput()

//______________________________________________________________________
//
void
DataArchiver::initializeOutput( const GridP& grid )
{
#ifdef HAVE_PIDX
  // Setup for PIDX
  if( savingAsPIDX() ) {
    if( m_pidx_requested_nth_rank == -1 ) {
      m_pidx_requested_nth_rank = m_loadBalancer->getNthRank();

      if( m_pidx_requested_nth_rank > 1 ) {
        proc0cout << "Input file requests output to be saved by every "
                  << m_pidx_requested_nth_rank << "th processor.\n"
                  << "  - However, setting output to every processor "
                  << "until a checkpoint is reached." << std::endl;
        m_loadBalancer->setNthRank( 1 );
        m_loadBalancer->possiblyDynamicallyReallocate( grid,
						       LoadBalancer::regrid );
        setSaveAsPIDX();
      }
    }
  }
#endif
}

//______________________________________________________________________
// to be called after problemSetup and initializeOutput get called
void
DataArchiver::restartSetup( Dir    & restartFromDir,
                            int      startTimeStep,
                            int      timestep,
                            double   time,
                            bool     fromScratch,
                            bool     removeOldDir )
{
  m_outputInitTimeStep = false;

  if( m_writeMeta && !fromScratch ) {
    // partial copy of dat files
    copyDatFiles( restartFromDir, m_dir, startTimeStep, timestep, removeOldDir );

    copySection( restartFromDir, m_dir, "index.xml", "restarts" );
    copySection( restartFromDir, m_dir, "index.xml", "variables" );
    copySection( restartFromDir, m_dir, "index.xml", "globals" );

    // partial copy of index.xml and timestep directories and
    // similarly for checkpoints
    copyTimeSteps(restartFromDir, m_dir, startTimeStep, timestep, removeOldDir);

    Dir checkpointsFromDir = restartFromDir.getSubdir("checkpoints");
    bool areCheckpoints = true;

    if (time > 0) {
      // the restart_merger doesn't need checkpoints, and calls this
      // with time=0.
      copyTimeSteps( checkpointsFromDir, m_checkpointsDir, startTimeStep,
                     timestep, removeOldDir, areCheckpoints );
      copySection( checkpointsFromDir, m_checkpointsDir, "index.xml", "variables" );
      copySection( checkpointsFromDir, m_checkpointsDir, "index.xml", "globals" );
    }

    if (removeOldDir) {
      // Try to remove the old dir...
      if( !Dir::removeDir( restartFromDir.getName().c_str() ) ) {
        // Something strange happened... let's test the filesystem...
        stringstream error_stream;          
        if( !testFilesystem( restartFromDir.getName(), error_stream, Parallel::getMPIRank() ) ) {

          cout << error_stream.str();
          cout.flush();

          // The file system just gave us some problems...
          printf( "WARNING: Filesystem check failed on processor %d\n", Parallel::getMPIRank() );
        }
        // Verify that "system works"
        int code = system( "echo how_are_you" );
        if( code != 0 ) {
          printf( "WARNING: test of system call failed\n" );
        }
      }
    }
  }
  else if( m_writeMeta ) { // Just add <restart from = ".." timestep = ".."> tag.
    copySection(restartFromDir, m_dir, "index.xml", "restarts");

    string iname = m_dir.getName()+"/index.xml";
    ProblemSpecP indexDoc = loadDocument(iname);

    if (timestep >= 0) {
      addRestartStamp(indexDoc, restartFromDir, timestep);
    }

    indexDoc->output(iname.c_str());
    //indexDoc->releaseDocument();
  }
} // end restartSetup()

//______________________________________________________________________
// This is called after problemSetup. It will copy the dat &
// checkpoint files to the new directory.  This also removes the
// global (dat) variables from the saveLabels variables
void
DataArchiver::postProcessUdaSetup(Dir& fromDir)
{
  //__________________________________
  // copy files
  // copy dat files and
  if (m_writeMeta) {
    m_fromDir = fromDir;
    copyDatFiles(fromDir, m_dir, 0, -1, false);
    copySection(fromDir,  m_dir, "index.xml", "globals");
    proc0cout << "*** Copied dat files to:   " << m_dir.getName() << "\n";
    
    // copy checkpoints
    Dir checkpointsFromDir = fromDir.getSubdir("checkpoints");
    Dir checkpointsToDir   = m_dir.getSubdir("checkpoints");
    string me = checkpointsFromDir.getName();
    if( validDir(me) ) {
      checkpointsToDir.remove( "index.xml", false);  // this file is created upstream when it shouldn't have
      checkpointsFromDir.copy( m_dir );
      proc0cout << "\n*** Copied checkpoints to: " << m_checkpointsDir.getName() << "\n";
      proc0cout << "    Only using 1 processor to copy so this will be slow for large checkpoint directories\n\n";
    }

    // copy input.xml.orig if it exists
    string there = m_dir.getName();
    string here  = fromDir.getName() + "/input.xml.orig";
    if ( validFile(here) ) {
      fromDir.copy("input.xml.orig", m_dir);     // use OS independent copy functions, needed by mira
      proc0cout << "*** Copied input.xml.orig to: " << there << "\n";
    }
    
    // copy the original ups file if it exists
    vector<string> ups;
    fromDir.getFilenamesBySuffix( "ups", ups );
    
    if ( ups.size() != 0 ) {
      fromDir.copy(ups[0], m_dir);              // use OS independent copy functions, needed by mira
      proc0cout << "*** Copied ups file ("<< ups[0]<< ") to: " << there << "\n";
    }
    proc0cout << "\n\n";
  }

  //__________________________________
  //
  // removed the global (dat) variables from the saveLabels
  string iname = fromDir.getName()+"/index.xml";
  ProblemSpecP indexDoc = loadDocument(iname);

  ProblemSpecP globals = indexDoc->findBlock("globals");
  if( globals != nullptr ) {

    ProblemSpecP variable = globals->findBlock("variable");
    while( variable != nullptr ) {
      string varname;

      if ( !variable->getAttribute("name", varname) ) {
        throw InternalError("global variable name attribute not found", __FILE__, __LINE__);
      }

      list<SaveNameItem>::iterator iter = m_saveLabelNames.begin();
      while ( iter != m_saveLabelNames.end() ) {
        if ( (*iter).labelName == varname ) {
          iter = m_saveLabelNames.erase(iter);
        }
        else {
          ++iter;
        }
      }
      variable = variable->findNextBlock("variable");
    }
  }
    
  //__________________________________
  //  Read in the timestep indicies from the restart uda and store them
  //  Use the indicies when creating the timestep directories
  ProblemSpecP ts_ps = indexDoc->findBlock("timesteps");
  ProblemSpecP ts    = ts_ps->findBlock("timestep");
  int timestep = -9;
  int count    = 1;
  
  while( ts != nullptr ) {
    ts->get(timestep);
    m_restartTimeStepIndicies[count] = timestep;
    
    ts = ts->findNextBlock("timestep");
    count ++;
  }
  
  m_restartTimeStepIndicies[0] = m_restartTimeStepIndicies[1];

  // Set checkpoint outputIntervals
  m_checkpointInterval = 0.0;
  m_checkpointTimeStepInterval = 0;
  m_checkpointWallTimeInterval = 0;
  m_nextCheckpointTimeStep  = SHRT_MAX;
  

  // output every timestep -- each timestep is transferring data
  m_outputInitTimeStep     = true;
  m_outputInterval         = 0.0;
  m_outputTimeStepInterval = 1;
  m_doPostProcessUda       = true;

} // end postProcessUdaSetup()

//______________________________________________________________________
//
void
DataArchiver::copySection( Dir& fromDir, Dir& toDir,
			   const string & filename, const string & section )
{
  // copy chunk labeled section between index.xml files
  string iname = fromDir.getName() + "/" +filename;
  ProblemSpecP indexDoc = loadDocument(iname);

  iname = toDir.getName() + "/" + filename;
  ProblemSpecP myIndexDoc = loadDocument(iname);
  
  ProblemSpecP sectionNode = indexDoc->findBlock(section);
  if( sectionNode != nullptr ) {
    ProblemSpecP newNode = myIndexDoc->importNode(sectionNode, true);
    
    // replace whatever was in the section previously
    ProblemSpecP mySectionNode = myIndexDoc->findBlock(section);
    if( mySectionNode != nullptr ) {
      myIndexDoc->replaceChild(newNode, mySectionNode);
    }
    else {
      myIndexDoc->appendChild(newNode);
    }
  }
  
  myIndexDoc->output(iname.c_str());

  //indexDoc->releaseDocument();
  //myIndexDoc->releaseDocument();
}

//______________________________________________________________________
//
void
DataArchiver::addRestartStamp(ProblemSpecP indexDoc, Dir& fromDir,
                              int timestep)
{
   // add restart history to restarts section
   ProblemSpecP restarts = indexDoc->findBlock("restarts");
   if( restarts == nullptr ) {
     restarts = indexDoc->appendChild("restarts");
   }

   // Restart from <dir> at timestep.
   ProblemSpecP restartInfo = restarts->appendChild( "restart" );
   restartInfo->setAttribute("from", fromDir.getName().c_str());
   
   ostringstream timestep_str;
   timestep_str << timestep;

   restartInfo->setAttribute("timestep", timestep_str.str().c_str());   
}

//______________________________________________________________________
//
void
DataArchiver::copyTimeSteps(Dir& fromDir, Dir& toDir, int startTimeStep,
                            int maxTimeStep, bool removeOld,
                            bool areCheckpoints /*=false*/)
{
   string old_iname = fromDir.getName()+"/index.xml";
   ProblemSpecP oldIndexDoc = loadDocument(old_iname);
   string iname = toDir.getName()+"/index.xml";
   ProblemSpecP indexDoc = loadDocument(iname);

   ProblemSpecP oldTimeSteps = oldIndexDoc->findBlock("timesteps");

   ProblemSpecP ts;
   if( oldTimeSteps != nullptr ) {
     ts = oldTimeSteps->findBlock("timestep");
   }

   // while we're at it, add restart information to index.xml
   if( maxTimeStep >= 0 ) {
     addRestartStamp(indexDoc, fromDir, maxTimeStep);
   }

   // create timesteps element if necessary
   ProblemSpecP timesteps = indexDoc->findBlock("timesteps");
   if( timesteps == nullptr ) {
      timesteps = indexDoc->appendChild("timesteps");
   }
   
   // copy each timestep 
   int timestep;
   while( ts != nullptr ) {
      ts->get(timestep);
      if (timestep >= startTimeStep &&
          (timestep <= maxTimeStep || maxTimeStep < 0)) {
         // copy the timestep directory over
         map<string,string> attributes;
         ts->getAttributes(attributes);

         string hrefNode = attributes["href"];
         if (hrefNode == "")
            throw InternalError("timestep href attribute not found",
				__FILE__, __LINE__);

         string::size_type href_pos = hrefNode.find_first_of("/");

         string href = hrefNode;
         if (href_pos != string::npos)
           href = hrefNode.substr(0, href_pos);
         
         //copy timestep directory
         Dir timestepDir = fromDir.getSubdir(href);
         if (removeOld)
            timestepDir.move(toDir);
         else
            timestepDir.copy(toDir);

         if (areCheckpoints)
            m_checkpointTimeStepDirs.push_back(toDir.getSubdir(href).getName());
         
         // add the timestep to the index.xml
         ostringstream timestep_str;
         timestep_str << timestep;
         ProblemSpecP newTS =
	   timesteps->appendElement("timestep", timestep_str.str().c_str());

	 map<string,string>::iterator iter;
         for ( iter = attributes.begin(); iter != attributes.end(); ++iter) {
           newTS->setAttribute((*iter).first, (*iter).second);
         }
         
      }
      ts = ts->findNextBlock("timestep");
   }

   // re-output index.xml
   indexDoc->output(iname.c_str());
   //indexDoc->releaseDocument();

   // we don't need the old document anymore...
   //oldIndexDoc->releaseDocument();

}

//______________________________________________________________________
//
void
DataArchiver::copyDatFiles(Dir& fromDir, Dir& toDir, int startTimeStep,
                           int maxTimeStep, bool removeOld)
{
   char buffer[1000];

   // find the dat file via the globals block in index.xml
   string iname = fromDir.getName()+"/index.xml";
   ProblemSpecP indexDoc = loadDocument(iname);

   ProblemSpecP globals = indexDoc->findBlock("globals");
   if( globals != nullptr ) {
      ProblemSpecP variable = globals->findBlock("variable");
      // copy data file associated with each variable
      while( variable != nullptr ) {
         map<string,string> attributes;
         variable->getAttributes(attributes);

         string hrefNode = attributes["href"];

         if (hrefNode == "") {
            throw InternalError("global variable href attribute not found",
				__FILE__, __LINE__);
         }
         const char* href = hrefNode.c_str();

         ifstream datFile((fromDir.getName()+"/"+href).c_str());
         if (!datFile) {
           throw InternalError("DataArchiver::copyDatFiles(): The file \"" + \
                               (fromDir.getName()+"/"+href) + \
                               "\" could not be opened for reading!",
			       __FILE__, __LINE__);
         }

         ofstream copyDatFile((toDir.getName()+"/"+href).c_str(), ios::app);
         if (!copyDatFile) {
           throw InternalError("DataArchiver::copyDatFiles(): The file \"" + \
                               (toDir.getName()+"/"+href) + \
                               "\" could not be opened for writing!",
			       __FILE__, __LINE__);
         }

         // copy up to maxTimeStep lines of the old dat file to the copy
         int timestep = startTimeStep;
         while (datFile.getline(buffer, 1000) &&
                (timestep < maxTimeStep || maxTimeStep < 0)) {
            copyDatFile << buffer << "\n";
            timestep++;
         }
         datFile.close();

         if (removeOld) 
            fromDir.remove(href, false);
         
         variable = variable->findNextBlock("variable");
      }
   }
   //indexDoc->releaseDocument();
}

//______________________________________________________________________
//
void
DataArchiver::createIndexXML(Dir& dir)
{
   ProblemSpecP rootElem = ProblemSpec::createDocument("Uintah_DataArchive");

   rootElem->appendElement("numberOfProcessors", d_myworld->nRanks());

   rootElem->appendElement("ParticlePosition", m_particlePositionName);
   
   string format = "uda";
   if ( m_outputFileFormat == PIDX ){
    format = "PIDX";
   }
   rootElem->appendElement( "outputFormat", format );
   
   ProblemSpecP metaElem = rootElem->appendChild("Meta");

   const char * logname = getenv( "LOGNAME" );
   if( logname ) {
     metaElem->appendElement( "username", logname );
   }
   else {
     // Some systems don't supply a logname
     // FIXME... is there a better way to find the username when
     // logname doesn't work?
     metaElem->appendElement( "username", "unknown" );
   }

   time_t t = time(nullptr) ;
   
   // Chop the newline character off the time string so that the Date
   // field will appear properly in the XML
   string time_string(ctime(&t));
   string::iterator end = time_string.end();
   --end;
   time_string.erase(end);
   metaElem->appendElement("date", time_string.c_str());
   metaElem->appendElement("endianness", endianness().c_str());
   metaElem->appendElement("nBits", (int)sizeof(unsigned long) * 8 );

   string iname = dir.getName() + "/index.xml";
   rootElem->output( iname.c_str() );
   //rootElem->releaseDocument();
}


//______________________________________________________________________
//
void
DataArchiver::finalizeTimeStep( const GridP & grid, 
                                SchedulerP  & sched,
                                bool          recompile /* = false */ )
{
  //this function should get called exactly once per timestep
  
  const int timeStep = m_application->getTimeStep();
  const double simTime = m_application->getSimTime();
  const double delT = m_application->getDelT();

  //  static bool wereSavesAndCheckpointsInitialized = false;
  if (dbg.active()) {
    dbg << " finalizeTimeStep,"
        << " time step = " << timeStep
        << " sim time = " << simTime
        << " delT= " << delT << "\n";
  }
  
  beginOutputTimeStep( grid );

  //__________________________________
  // some changes here - we need to redo this if we add a material, or
  // if we schedule output on the initialization timestep (because
  // there will be new computes on subsequent timestep) or if there is
  // a component switch or a new level in the grid - BJW
  if (((delT != 0.0 || m_outputInitTimeStep) &&
       !m_wereSavesAndCheckpointsInitialized) ||
      
      m_switchState ||
      
      grid->numLevels() != m_numLevelsInOutput) {
    
    // Skip the initialization timestep (normally, anyway) for this
    // because it needs all computes to be set to find the save labels    
    if( m_outputInterval > 0.0 || m_outputTimeStepInterval > 0 ) {
      initSaveLabels(sched, delT == 0.0);
     
      if (!m_wereSavesAndCheckpointsInitialized && delT != 0.0) {
        indexAddGlobals(); // add saved global (reduction) variables to index.xml
      }
    }
    
    // This assumes that the TaskGraph doesn't change after the second
    // timestep and will need to change if the TaskGraph becomes dynamic. 
    // We also need to do this again if this is the initial timestep
    if (delT != 0.0) {
      m_wereSavesAndCheckpointsInitialized = true;
    
      // Can't do checkpoints on init timestep....
      if (m_checkpointInterval > 0.0 ||
	  m_checkpointTimeStepInterval > 0 ||
	  m_checkpointWallTimeInterval > 0) {
        initCheckpoints(sched);
      }
    }
  }
  
  m_numLevelsInOutput = grid->numLevels();
  
#if SCI_ASSERTION_LEVEL >= 2
  m_outputCalled.clear();
  m_outputCalled.resize(m_numLevelsInOutput, false);
  m_checkpointCalled.clear();
  m_checkpointCalled.resize(m_numLevelsInOutput, false);
  m_checkpointReductionCalled = false;
#endif
}

//______________________________________________________________________
//  Schedule output tasks for the grid variables, particle variables and reduction variables
void
DataArchiver::sched_allOutputTasks( const GridP      & grid, 
                                          SchedulerP & sched,
                                          bool         recompile /* = false */ )
{
  if (dbg.active()) {
    dbg << "  sched_allOutputTasks \n";
  }
  
  // Don't schedule more tasks unless recompiling.
  if ( !recompile ) {
    return;
  }

  const double delT = m_application->getDelT();

  //__________________________________
  //  Reduction Variables
  // Schedule task to dump out reduction variables at every timestep
  
  if( (m_outputInterval  > 0.0 || m_outputTimeStepInterval  > 0) &&
      (delT != 0.0 || m_outputInitTimeStep)) {
    
    Task* task = scinew Task( "DataArchiver::outputReductionVars",
			   this, &DataArchiver::outputReductionVars );

    // task->requires( Task::OldDW, m_simTimeLabel );
    task->requires( Task::OldDW, m_delTLabel );
    
    for( int i=0; i<(int)m_saveReductionLabels.size(); ++i) {
      SaveItem& saveItem = m_saveReductionLabels[i];
      const VarLabel* var = saveItem.label;
      
      const MaterialSubset* matls = saveItem.getMaterialSubset(0);
      task->requires( Task::NewDW, var, matls, true );
    }
    
    sched->addTask(task, nullptr, nullptr);
    
    if (dbg.active()) {
      dbg << "  scheduled output tasks (reduction variables)\n";
    }

    if ( delT != 0.0 || m_outputInitTimeStep ) {
      scheduleOutputTimeStep( m_saveLabels, grid, sched, false );
    }
  }
  
  //__________________________________
  //  Schedule Checkpoint (reduction variables)
  if (delT != 0.0 && m_checkpointCycle > 0 &&
      ( m_checkpointInterval > 0 ||
	m_checkpointTimeStepInterval > 0 ||
	m_checkpointWallTimeInterval > 0 ) ) {
    
    // output checkpoint timestep
    Task* task = scinew Task( "DataArchiver::outputVariables (CheckpointReduction)",
			      this, &DataArchiver::outputVariables, CHECKPOINT_REDUCTION );

    // task->requires( Task::OldDW, m_timeStepLabel);
    
    for( int i = 0; i < (int) m_checkpointReductionLabels.size(); i++ ) {
      SaveItem& saveItem = m_checkpointReductionLabels[i];
      const VarLabel* var = saveItem.label;
      const MaterialSubset* matls = saveItem.getMaterialSubset(0);
      
      task->requires(Task::NewDW, var, matls, true);
    }
    sched->addTask(task, nullptr, nullptr);
    
    if (dbg.active()) {
      dbg << "  scheduled output tasks (checkpoint variables)\n";
    }
    
    scheduleOutputTimeStep( m_checkpointLabels, grid, sched, true );
  }
  
#if HAVE_PIDX
  if ( m_outputFileFormat == PIDX ) {
    /*  Create PIDX communicators (one communicator per AMR level) */

    // FIXME: It appears that this is called 3 or more times before
    //        timestep 0... why is this the case?  Doesn't hurt
    //        anything, but we should figure out why...

    // ARS - It can get called multiple times when regridding.
    
    dbg << "  Creating communicatore per AMR level (required for PIDX)\n";
    createPIDXCommunicator( m_checkpointLabels,  grid, sched, true );
  }
#endif  

} // end sched_allOutputTasks()


//______________________________________________________________________
//
void
DataArchiver::beginOutputTimeStep( const GridP& grid )
{
  const int timeStep = m_application->getTimeStep();
  const double simTime = m_application->getSimTime();
  const double delT = m_application->getDelT();

  if (dbg.active()) {
    dbg << "    beginOutputTimeStep\n";
  }

  // Do *not* update the next values here as the original values are
  // needed to compare with if there is a timestep restart.  See
  // reEvaluateOutputTimeStep

  // Check for an output.
  m_isOutputTimeStep =
    // Output based on the simulation time.
    ( ((m_outputInterval > 0.0 &&
	(delT != 0.0 || m_outputInitTimeStep)) &&
       (simTime + delT >= m_nextOutputTime) ) ||
      
      // Output based on the timestep interval.
      ((m_outputTimeStepInterval > 0 &&
	(delT != 0.0 || m_outputInitTimeStep)) &&
       (timeStep >= m_nextOutputTimeStep)) ||
      
      // Output based on the being the last timestep.
      (m_outputLastTimeStep && m_maybeLastTimeStep) );

  // Create the output timestep directories
  if( m_isOutputTimeStep && m_outputFileFormat != PIDX ) {
    makeTimeStepDirs( m_dir, m_saveLabels, grid, &m_lastTimeStepLocation );
  }
  
  // Check for a checkpoint.
  m_isCheckpointTimeStep =
    // Checkpoint based on the simulation time.
    ( (m_checkpointInterval > 0.0 &&
       (simTime + delT) >= m_nextCheckpointTime) ||
      
      // Checkpoint based on the timestep interval.
      (m_checkpointTimeStepInterval > 0 &&
       timeStep >= m_nextCheckpointTimeStep) ||
      
      // Checkpoint based on the being the last timestep.
      (m_checkpointLastTimeStep && m_maybeLastTimeStep) );    

  // Checkpoint based on the being the wall time.
  if( m_checkpointWallTimeInterval > 0 ) {

    if( m_elapsedWallTime >= m_nextCheckpointWallTime )
      m_isCheckpointTimeStep = true;	
  }
  
  // Create the output checkpoint directories
  if( m_isCheckpointTimeStep ) {
    
    string timestepDir;
    makeTimeStepDirs( m_checkpointsDir, m_checkpointLabels, grid, &timestepDir );
    
    string iname = m_checkpointsDir.getName() + "/index.xml";

    ProblemSpecP index;
    
    if (m_writeMeta) {
      index = loadDocument(iname);
      
      // store a back up in case it dies while writing index.xml
      string ibackup_name = m_checkpointsDir.getName()+"/index_backup.xml";
      index->output(ibackup_name.c_str());
    }

    m_checkpointTimeStepDirs.push_back(timestepDir);
    
    if ((int)m_checkpointTimeStepDirs.size() > m_checkpointCycle) {
      if (m_writeMeta) {
        // remove reference to outdated checkpoint directory from the
        // checkpoint index
        ProblemSpecP ts = index->findBlock("timesteps");
        ProblemSpecP temp = ts->getFirstChild();
        ts->removeChild(temp);

        index->output(iname.c_str());
        
        // remove out-dated checkpoint directory
        Dir expiredDir(m_checkpointTimeStepDirs.front());

        // Try to remove the expired checkpoint directory...
        if( !Dir::removeDir( expiredDir.getName().c_str() ) ) {
          // Something strange happened... let's test the filesystem...
          stringstream error_stream;          
          if( !testFilesystem( expiredDir.getName(), error_stream, Parallel::getMPIRank() ) ) {
            cout << error_stream.str();
            cout.flush();
            // The file system just gave us some problems...
            printf( "WARNING: Filesystem check failed on processor %d\n", Parallel::getMPIRank() );
          }
        }
      }
      m_checkpointTimeStepDirs.pop_front();
    }
    //if (m_writeMeta)
    //index->releaseDocument();
  }

  if (dbg.active()) {
    dbg << "    write output timestep (" << m_isOutputTimeStep << ")" << std::endl
        << "    write CheckPoints (" << m_isCheckpointTimeStep << ")" << std::endl
        << "    end\n";
  }
  
} // end beginOutputTimeStep

//______________________________________________________________________
//
void
DataArchiver::makeTimeStepDirs(       Dir                            & baseDir,
                                      vector<DataArchiver::SaveItem> & saveLabels ,
                                const GridP                          & grid,
                                      string                         * pTimeStepDir /* passed back */ )
{
  const int timeStep = m_application->getTimeStep();

  int numLevels = grid->numLevels();
  // time should be currentTime+delt
  
  int dir_timestep = getTimeStepTopLevel();

  if (dbg.active()) {
    dbg << "      makeTimeStepDirs for timestep: " << timeStep
        << " dir_timestep: " << dir_timestep<< "\n";
  }
  
  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << dir_timestep;
  *pTimeStepDir = baseDir.getName() + "/" + tname.str();

  //__________________________________
  // Create the directory for this time step, if necessary
  // It is not gurantteed that the rank holding m_writeMeta will call 
  // outputTimstep to create dir before another rank begin to output data.
  // A race condition happens when a rank executes output task and
  // the rank holding m_writeMeta is still compiling task graph. 
  // So every rank should try to create dir.
  //if(m_writeMeta) {

  Dir tdir = baseDir.createSubdirPlus(tname.str());
  
  // Create the directory for this level, if necessary
  for( int l = 0; l < numLevels; l++ ) {
    ostringstream lname;
    lname << "l" << l;
    Dir ldir = tdir.createSubdirPlus( lname.str() );
    
    createPIDX_dirs( saveLabels,ldir );
  }
}


//______________________________________________________________________
//
void
DataArchiver::reevaluate_OutputCheckPointTimeStep( const double simTime,
						   const double delT )
{
  if (dbg.active()) {
    dbg << "  reevaluate_OutputCheckPointTimeStep() begin\n";
  }

  // Call this on a timestep restart. If lowering the delt goes
  // beneath the threshold, cancel the output and/or checkpoint
  // timestep

  if (m_isOutputTimeStep && m_outputInterval > 0.0 ) {
    if (simTime+delT < m_nextOutputTime)
      m_isOutputTimeStep = false;
  }
  
  if (m_isCheckpointTimeStep && m_checkpointInterval > 0.0) {
    if (simTime+delT < m_nextCheckpointTime) {
      m_isCheckpointTimeStep = false;    
      m_checkpointTimeStepDirs.pop_back();
    }
  }

#if SCI_ASSERTION_LEVEL >= 2
  m_outputCalled.clear();
  m_outputCalled.resize(m_numLevelsInOutput, false);
  m_checkpointCalled.clear();
  m_checkpointCalled.resize(m_numLevelsInOutput, false);
  m_checkpointReductionCalled = false;
#endif

  if (dbg.active()) {
    dbg << "  reevaluate_OutputCheckPointTimeStep() end\n";
  }
}

//______________________________________________________________________
//
void
DataArchiver::findNext_OutputCheckPointTimeStep( const bool restart,
						 const GridP& grid )
{
  if (dbg.active()) {
    dbg << "  findNext_OutputCheckPoint_TimeStep() begin\n";
  }

  const int timeStep = m_application->getTimeStep();
  const double simTime = m_application->getSimTime();
  const double delT = m_application->getNextDelT();

  if( restart )
  {
    // Output based on the simulaiton time.
    if( m_outputInterval > 0.0 ) {
      m_nextOutputTime = ceil(simTime / m_outputInterval) * m_outputInterval;
    }
    // Output based on the time step.
    else if( m_outputTimeStepInterval > 0 ) {
      m_nextOutputTimeStep =
	(timeStep/m_outputTimeStepInterval) * m_outputTimeStepInterval + 1;

      while( m_nextOutputTimeStep <= timeStep ) {
	m_nextOutputTimeStep += m_outputTimeStepInterval;
      }
    }
   
    // Checkpoint based on the simulaiton time.
    if( m_checkpointInterval > 0.0 ) {
      m_nextCheckpointTime =
	ceil(simTime / m_checkpointInterval) * m_checkpointInterval;
    }
    // Checkpoint based on the time step.
    else if( m_checkpointTimeStepInterval > 0 ) {
      m_nextCheckpointTimeStep =
	(timeStep / m_checkpointTimeStepInterval) *
	m_checkpointTimeStepInterval + 1;
      while( m_nextCheckpointTimeStep <= timeStep ) {
	m_nextCheckpointTimeStep += m_checkpointTimeStepInterval;
      }
    }
    // Checkpoint based on the wall time.
    else if( m_checkpointWallTimeInterval > 0 ) {
      m_nextCheckpointWallTime = m_elapsedWallTime + m_checkpointWallTimeInterval;
    }
  }
  
  // If this timestep was an output/checkpoint timestep, determine
  // when the next one will be.

  // Do *not* do this step in beginOutputTimeStep because the original
  // values are needed to compare with if there is a timestep restart.
  // See reEvaluateOutputTimeStep

  // When outputing/checkpointing using the simulation or wall time
  // check to see if the simulation or wall time went past more than
  // one interval. If so adjust accordingly.

  // Note - it is not clear why but when outputing/checkpointing using
  // time steps the mod function must also be used. This does not
  // affect most simulations except when there are multiple UPS files
  // such as when components are switched. For example:
  // StandAlone/inputs/UCF/Switcher/switchExample3.ups

  else if( m_isOutputTimeStep ) {
    
    // Output based on the simulaiton time.
    if( m_outputInterval > 0.0 ) {
      if( simTime >= m_nextOutputTime ) {
        m_nextOutputTime +=
	  floor( (simTime - m_nextOutputTime) / m_outputInterval ) *
	  m_outputInterval + m_outputInterval;
      }
    }
    // Output based on the time step.
    else if( m_outputTimeStepInterval > 0 ) {
      if( timeStep >= m_nextOutputTimeStep )  {
        m_nextOutputTimeStep +=
	  ( (timeStep - m_nextOutputTimeStep) / m_outputTimeStepInterval ) *
	  m_outputTimeStepInterval + m_outputTimeStepInterval;
      }
    }
  }

  if( m_isCheckpointTimeStep ) {
    // Checkpoint based on the simulaiton time.
    if( m_checkpointInterval > 0.0 ) {
      if( simTime >= m_nextCheckpointTime ) {
        m_nextCheckpointTime +=
	  floor( (simTime - m_nextCheckpointTime) / m_checkpointInterval ) *
	  m_checkpointInterval + m_checkpointInterval;
      }
    }
    // Checkpoint based on the time step.
    else if( m_checkpointTimeStepInterval > 0 ) {
      if( timeStep >= m_nextCheckpointTimeStep ) {
        m_nextCheckpointTimeStep +=
	  ( (timeStep - m_nextCheckpointTimeStep) /
	    m_checkpointTimeStepInterval ) *
	  m_checkpointTimeStepInterval + m_checkpointTimeStepInterval;
      }
    }

    // Checkpoint based on the wall time.
    else if( m_checkpointWallTimeInterval > 0 ) {

      if( m_elapsedWallTime >= m_nextCheckpointWallTime ) {
        m_nextCheckpointWallTime +=
	  floor( (m_elapsedWallTime - m_nextCheckpointWallTime) /
		 m_checkpointWallTimeInterval ) *
	  m_checkpointWallTimeInterval + m_checkpointWallTimeInterval;
      }
    }
  }

  // When saving via PIDX one needs to predict if one is going to do a
  // checkpoint. These are the exact same checks as done in
  // beginOutputTimeStep
#ifdef HAVE_PIDX
  if( savingAsPIDX() ) {

    // Check for a checkpoint.
    m_pidx_checkpointing =
      // Checkpoint based on the simulation time.
      ( (m_checkpointInterval > 0.0 &&
	 (simTime + delT) >= m_nextCheckpointTime) ||
	
	// Checkpoint based on the timestep interval.
	(m_checkpointTimeStepInterval > 0 &&
	 timeStep >= m_nextCheckpointTimeStep) ||
	
	// Checkpoint based on the being the last timestep.
	(m_checkpointLastTimeStep && m_maybeLastTimeStep) );    
    
    // Checkpoint based on the being the wall time.
    if( m_checkpointWallTimeInterval > 0 ) {
      
      if( m_elapsedWallTime >= m_nextCheckpointWallTime )
	m_pidx_checkpointing = true;	
    }
    
    // Checkpointing
    if( m_pidx_checkpointing ) {
      
      if( m_pidx_requested_nth_rank > 1 ) {
	proc0cout << "This is a checkpoint time step (" << timeStep
		  << ") - need to recompile with nth proc set to: "
		  << m_pidx_requested_nth_rank << std::endl;
	
	m_loadBalancer->setNthRank( m_pidx_requested_nth_rank );
	m_loadBalancer->possiblyDynamicallyReallocate( grid,
						       LoadBalancer::regrid );
	setSaveAsUDA();
	m_pidx_need_to_recompile = true;
      }
    }

    // Check for an output which may need to be postponed

    // Output based on the simulation time.
    if( ((m_outputInterval > 0.0 &&
	  (delT != 0.0 || m_outputInitTimeStep)) &&
	 (simTime + delT >= m_nextOutputTime) ) ||
      
	// Output based on the timestep interval.
	((m_outputTimeStepInterval > 0 &&
	  (delT != 0.0 || m_outputInitTimeStep)) &&
	 (timeStep >= m_nextOutputTimeStep)) ||
	
	// Output based on the being the last timestep.
	(m_outputLastTimeStep && m_maybeLastTimeStep) ) {

      proc0cout << "This is an output time step: " << timeStep << ".  ";

      // If this is also a checkpoint time step postpone the output
      if( m_pidx_need_to_recompile ) {
	postponeNextOutputTimeStep();

	proc0cout << "   Postposing as it is also a checkpoint time step.";
      }

      proc0cout << std::endl;
    }
  }
#endif
  
  if (dbg.active()) {
    dbg << "  " << std::setprecision(15) << timeStep
        << "  " << std::setprecision(15) << simTime << "  " << "\n"
        << "    is output timestep: " << m_isOutputTimeStep
        << "       output interval: " << std::setprecision(15) << m_outputInterval
        << "  next output sim time: " << std::setprecision(15) << m_nextOutputTime
        << "       output timestep: " << m_outputTimeStepInterval
        << "  next output timestep: " << m_nextOutputTimeStep << "\n"

        << "    is checkpoint timestep: " << m_isCheckpointTimeStep
        << "       checkpoint interval: " << std::setprecision(15) << m_checkpointInterval
        << "  next checkpoint sim time: " << std::setprecision(15) << m_nextCheckpointTime
        << "       checkpoint timestep: " << m_checkpointTimeStepInterval
        << "  next checkpoint timestep: " << m_nextCheckpointTimeStep
        << "  next checkpoint WallTime: " << std::setprecision(15) << m_nextCheckpointWallTime << "\n";
    
    dbg << "  findNext_OutputCheckPoint_TimeStep() end\n";
  }

} // end findNext_OutputCheckPoint_TimeStep()


//______________________________________________________________________
//  update the xml files (index.xml, timestep.xml, 
void
DataArchiver::writeto_xml_files( const GridP& grid )
{
#ifdef HAVE_PIDX
  // For PIDX only save timestep.xml when checkpointing.  Normal
  // time step dumps using PIDX do not need to write the xml
  // information.      
  if( savingAsPIDX() && !m_pidx_checkpointing )
    return;
#endif
    
  const double simTime = m_application->getSimTime();
  const double delT = m_application->getDelT();

  Timers::Simple timer;
  timer.start();
  
  if (dbg.active()) {
    dbg << "  writeto_xml_files() begin\n";
  }

  if( !m_isCheckpointTimeStep && !m_isOutputTimeStep ) {
    if (dbg.active()) {
      dbg << "   This is not an output (or checkpoint) timestep, so just returning...\n";
    }
    return;
  }
  
  //__________________________________
  //  Writeto XML files
  // to check for output nth proc
  int dir_timestep = getTimeStepTopLevel();
  
  // start dumping files to disk
  vector<Dir*> baseDirs;
  if ( m_isOutputTimeStep ) {
    baseDirs.push_back( &m_dir );
  }    
  if ( m_isCheckpointTimeStep ) {
    baseDirs.push_back( &m_checkpointsDir );
  }

  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << dir_timestep;

  for (int i = 0; i < static_cast<int>( baseDirs.size() ); ++i) {
    // to save the list of vars. up to 2, since in checkpoints, there
    // are two types of vars
    vector< vector<SaveItem>* > savelist; 
    
    // Reference this timestep in index.xml
    if(m_writeMeta) {
      bool hasGlobals = false;

      if ( baseDirs[i] == &m_dir ) {
        savelist.push_back( &m_saveLabels );
      }
      else if ( baseDirs[i] == &m_checkpointsDir ) {
        hasGlobals = m_checkpointReductionLabels.size() > 0;
        savelist.push_back( &m_checkpointLabels );
        savelist.push_back( &m_checkpointReductionLabels );
      }
      else {
        throw "DataArchiver::writeto_xml_files(): Unknown directory!";
      }

      string iname = baseDirs[i]->getName()+"/index.xml";
      ProblemSpecP indexDoc = loadDocument(iname);

      // if this timestep isn't already in index.xml, add it in
      if( indexDoc == nullptr ) {
        continue; // output timestep but no variables scheduled to be saved.
      }
      ASSERT( indexDoc != nullptr );

      //__________________________________
      // output data pointers
      for (unsigned j = 0; j < savelist.size(); ++j) {
        string variableSection = savelist[j] == &m_checkpointReductionLabels ? "globals" : "variables";
        ProblemSpecP vs = indexDoc->findBlock(variableSection);
        if( vs == nullptr ) {
          vs = indexDoc->appendChild(variableSection.c_str());
        }
        for (unsigned k = 0; k < savelist[j]->size(); ++k) {
          const VarLabel* var = (*savelist[j])[k].label;
          bool found=false;
          
          for(ProblemSpecP n = vs->getFirstChild(); n != nullptr; n=n->getNextSibling()) {
            if(n->getNodeName() == "variable") {
              map<string,string> attributes;
              n->getAttributes(attributes);
              string varname = attributes["name"];
          
              if(varname == ""){
                throw InternalError("varname not found", __FILE__, __LINE__);
              }
              
              if(varname == var->getName()) {
                found=true;
                break;
              }
            }
          }
          if(!found) {
            ProblemSpecP newElem = vs->appendChild("variable");
            newElem->setAttribute("type", TranslateVariableType( var->typeDescription()->getName(), 
                                                                 baseDirs[i] != &m_dir ) );
            newElem->setAttribute("name", var->getName());
          }
        }
      }

      //__________________________________
      // Check if it's the first checkpoint timestep by checking if
      // the "timesteps" field is in checkpoints/index.xml.  If it is
      // then there exists a timestep.xml file already.  Use this
      // below to change information in input.xml...
      bool firstCheckpointTimeStep = false;
      
      ProblemSpecP ts = indexDoc->findBlock("timesteps");
      if( ts == nullptr ) {
        ts = indexDoc->appendChild("timesteps");
        firstCheckpointTimeStep = (&m_checkpointsDir == baseDirs[i]);
      }
      bool found=false;
      for(ProblemSpecP n = ts->getFirstChild(); n != nullptr; n=n->getNextSibling()) {
        if(n->getNodeName() == "timestep") {
          int readtimestep;
          
          if(!n->get(readtimestep)){
            throw InternalError("Error parsing timestep number", __FILE__, __LINE__);
          }
          if(readtimestep == dir_timestep) {
            found=true;
            break;
          }
        }
      }

      //__________________________________
      // add timestep info
      if(!found) {
        
        string timestepindex = tname.str()+"/timestep.xml";      
        
        ostringstream value, timeVal, deltVal;
        value << dir_timestep;
        ProblemSpecP newElem = ts->appendElement( "timestep",value.str().c_str() );
        newElem->setAttribute( "href",     timestepindex.c_str() );
        timeVal << std::setprecision(17) << simTime;// + delT;
        newElem->setAttribute( "time",     timeVal.str() );
        deltVal << std::setprecision(17) << delT;
        newElem->setAttribute( "oldDelt",  deltVal.str() );
      }
     
      indexDoc->output(iname.c_str());
      //indexDoc->releaseDocument();

      // make a timestep.xml file for this time step we need to do it
      // here in case there is a timestesp restart Break out the
      // <Grid> and <Data> section of the DOM tree into a separate
      // grid.xml file which can be created quickly and use less
      // memory using the xmlTextWriter functions (streaming output)

      ProblemSpecP rootElem = ProblemSpec::createDocument( "Uintah_timestep" );

      // Create a metadata element to store the per-timestep endianness
      ProblemSpecP metaElem = rootElem->appendChild("Meta");

      metaElem->appendElement("endianness", endianness().c_str());
      metaElem->appendElement("nBits", (int)sizeof(unsigned long) * 8 );
      metaElem->appendElement("numProcs", d_myworld->nRanks());

      // TimeStep information
      ProblemSpecP timeElem = rootElem->appendChild("Time");
      timeElem->appendElement("timestepNumber", dir_timestep);
      timeElem->appendElement("currentTime", simTime);// + delT);
      timeElem->appendElement("oldDelt", delT);

      //__________________________________
      // Output grid section:
      //
      // With AMR, we're not guaranteed that a rank has work on a
      // given level.  Quick check to see that, so we don't create a
      // node that points to no data.

      string grim_path = baseDirs[i]->getName() + "/" + tname.str() + "/";

#if XML_TEXTWRITER

      writeGridTextWriter( hasGlobals, grim_path, grid );
#else
      // Original version:
      writeGridOriginal( hasGlobals, grid, rootElem );

      // Binary Grid version:
      // writeGridBinary( hasGlobals, grim_path, grid );
#endif
      // Add the <Materials> section to the timestep.xml
      GeometryPieceFactory::resetGeometryPiecesOutput();

      // output each components output Problem spec
      m_application->outputProblemSpec( rootElem );

      outputProblemSpec( rootElem );

      // write out the timestep.xml file
      string name = baseDirs[i]->getName()+"/"+tname.str()+"/timestep.xml";
      rootElem->output( name.c_str() );
      //__________________________________
      // output input.xml & input.xml.orig

      // a small convenience to the user who wants to change things
      // when he restarts let him know that some information to change
      // will need to be done in the timestep.xml file instead of the
      // input.xml file.  Only do this once, though.
      
      if (firstCheckpointTimeStep) {
        // loop over the blocks in timestep.xml and remove them from
        // input.xml, with some exceptions.
        string inputname = m_dir.getName()+"/input.xml";
        ProblemSpecP inputDoc = loadDocument(inputname);
        inputDoc->output((inputname + ".orig").c_str());

        for (ProblemSpecP ps = rootElem->getFirstChild(); ps != nullptr; ps = ps->getNextSibling()) {
          string nodeName = ps->getNodeName();
          
          if (nodeName == "Meta" || nodeName == "Time" ||
	      nodeName == "Grid" || nodeName == "Data") {
            continue;
          }
          
          // find and replace the node 
          ProblemSpecP removeNode = inputDoc->findBlock(nodeName);
          if (removeNode != nullptr) {
            string comment = "The node " + nodeName + " has been removed.  Its original values are\n"
              "    in input.xml.orig, and values used for restarts are in checkpoints/t*****/timestep.xml";
            ProblemSpecP commentNode = inputDoc->makeComment(comment);
            inputDoc->replaceChild(removeNode, commentNode);
          }
        }
        inputDoc->output(inputname.c_str());
        //inputDoc->releaseDocument();
      }
      //rootElem->releaseDocument();
      
      //__________________________________
      // copy the component sections of timestep.xml.
      if( m_doPostProcessUda ) {
        copy_outputProblemSpec( m_fromDir, m_dir );
      }
    }
  }  // loop over baseDirs

  double myTime = timer().seconds();
  (*m_runTimeStats)[XMLIOTime] += myTime;
  (*m_runTimeStats)[TotalIOTime ] += myTime;

  if (dbg.active()) {
    dbg << "  end\n";
  }
}

//______________________________________________________________________
//  update the xml file index.xml with any in-situ modified variables.
void
DataArchiver::writeto_xml_files( std::map< std::string,
				 std::pair<std::string,
				 std::string> > &modifiedVars )
{
#ifdef HAVE_VISIT
//   if( isProc0_macro && m_sharedState->getVisIt() && modifiedVars.size() )
  {
    dbg << "  writeto_xml_files() begin\n";

    //__________________________________
    //  Writeto XML files
    // to check for output nth proc
    int dir_timestep = getTimeStepTopLevel();
  
    string iname = m_dir.getName()+"/index.xml";

    ProblemSpecP indexDoc = loadDocument(iname);
    
    string inSituSection("InSitu");
	  
    ProblemSpecP iss = indexDoc->findBlock(inSituSection);
	  
    if(iss.get_rep() == nullptr) {
      iss = indexDoc->appendChild(inSituSection.c_str());
    }

    const int timeStep = m_application->getTimeStep();

    std::stringstream timeStepStr;
    timeStepStr << timeStep + 1;
      
    // Report on the modiied variables. 
    std::map<std::string,std::pair<std::string, std::string> >::iterator iter;
    for (iter = modifiedVars.begin(); iter != modifiedVars.end(); ++iter) {
      proc0cout << "Visit libsim - For time step " << timeStepStr.str() << " "
		<< "the variable "         << iter->first << " "
		<< "will be changed from " << iter->second.first << " "
		<< "to "                   << iter->second.second << ". "
		<< std::endl;

      ProblemSpecP newElem = iss->appendChild("modifiedVariable");

      newElem->setAttribute("timestep", timeStepStr.str());
      newElem->setAttribute("name",     iter->first);
      newElem->setAttribute("oldValue", iter->second.first);
      newElem->setAttribute("newValue", iter->second.second);
    }

    indexDoc->output(iname.c_str());
    // indexDoc->releaseDocument();
    
    modifiedVars.clear();
  }
#endif  
}

void
DataArchiver::writeGridBinary( const bool hasGlobals, const string & grim_path, const GridP & grid )
{
  // Originally the <Grid> was saved in XML.  While this makes it very
  // human readable, for large runs (10K+ patches), the (original)
  // timestep.xml file, and now the grid.xml file become huge, taking
  // a significant amount of disk space, and time to generate/save.
  // To fix this problem, we will now write the data out in binary.
  // The original <Grid> data looks like this:
  //
  // <Grid>
  //   <numLevels>1</numLevels>
  //   <Level>
  //    <numPatches>4</numPatches>
  //    <totalCells>8820</totalCells>
  //    <extraCells>[1,1,1]</extraCells>
  //    <anchor>[0,0,0]</anchor>
  //    <id>0</id>
  //    <cellspacing>[0.025000000000000001,0.025000000000000001,0.049999999999999996]</cellspacing>
  //    <Patch>
  //     <id>0</id>
  //     <proc>0</proc>
  //     <lowIndex>[-1,-1,-1]</lowIndex>
  //     <highIndex>[20,20,4]</highIndex>
  //     <interiorLowIndex>[0,0,0]</interiorLowIndex>
  //     <interiorHighIndex>[20,20,3]</interiorHighIndex>
  //     <nnodes>2646</nnodes>
  //     <lower>[-0.025000000000000001,-0.025000000000000001,-0.049999999999999996]</lower>
  //     <upper>[0.5,0.5,0.19999999999999998]</upper>
  //     <totalCells>2205</totalCells>
  //    </Patch>
  //   </Level>
  // </Grid>

  FILE * fp;
  int marker = DataArchive::GRID_MAGIC_NUMBER;

  string grim_filename = grim_path + "grid.xml";

  fp = fopen( grim_filename.c_str(), "wb" );
  fwrite( &marker, sizeof(int), 1, fp );

  // NUmber of Levels
  int numLevels = grid->numLevels();
  fwrite( &numLevels, sizeof(int), 1, fp );

  vector< vector<bool> > procOnLevel( numLevels );

  for( int lev = 0; lev < numLevels; lev++ ) {

    LevelP level = grid->getLevel( lev );

    int    num_patches = level->numPatches();
    long   num_cells   = level->totalCells();
    IntVector eciv = level->getExtraCells();
    int  * extra_cells = eciv.get_pointer();
    double anchor[3];
    Point  anchor_pt = level->getAnchor();
    int    id = level->getID();
    double cell_spacing[3];
    Vector cell_spacing_vec = level->dCell();
    IntVector pbiv = level->getPeriodicBoundaries();
    int     * period = pbiv.get_pointer();
    
    anchor[0] = anchor_pt.x();
    anchor[1] = anchor_pt.y();
    anchor[2] = anchor_pt.z();

    cell_spacing[0] = cell_spacing_vec.x();
    cell_spacing[1] = cell_spacing_vec.y();
    cell_spacing[2] = cell_spacing_vec.z();

    fwrite( &num_patches, sizeof(int),    1, fp );  // Number of Patches -  100
    fwrite( &num_cells,   sizeof(long),   1, fp );  // Number of Cells   - 8000
    fwrite( extra_cells,  sizeof(int),    3, fp );  // Extra Cell Info - [1,1,1]
    fwrite( anchor,       sizeof(double), 3, fp );  // Anchor Info     - [0,0,0]
    fwrite( period,       sizeof(int),    3, fp );  // 
    fwrite( &id,          sizeof(int),    1, fp );  // ID of Level     - 0
    fwrite( cell_spacing, sizeof(double), 3, fp );  // Cell Spacing - [0.1,0.1,0.1]

    procOnLevel[ lev ].resize( d_myworld->nRanks() );

    // Iterate over patches.
    Level::const_patch_iterator iter;
    for( iter = level->patchesBegin(); iter != level->patchesEnd(); ++iter ) {
      const Patch* patch = *iter;

      int       patch_id   = patch->getID();
      int       rank_id    = m_loadBalancer->getOutputRank( patch );

      int proc = m_loadBalancer->getOutputRank( patch );
      procOnLevel[ lev ][ proc ] = true;

      IntVector ecliiv  = patch->getExtraCellLowIndex();
      IntVector echiiv = patch->getExtraCellHighIndex();
      int     * low_index  = ecliiv.get_pointer();
      int     * high_index = echiiv.get_pointer();

      // Interior indices
      IntVector cliiv = patch->getCellLowIndex();
      IntVector chiiv = patch->getCellHighIndex();
      int    * i_low_index  = cliiv.get_pointer();
      int    * i_high_index = chiiv.get_pointer();

      int      num_nodes    = patch->getNumExtraNodes();

      Box      box          = patch->getExtraBox();
      
      double         lower[3];
      const Point  & lower_pt = box.lower();
      lower[0] = lower_pt.x();
      lower[1] = lower_pt.y();
      lower[2] = lower_pt.z();
      
      double         upper[3];
      const Point  & upper_pt = box.upper();
      upper[0] = upper_pt.x();
      upper[1] = upper_pt.y();
      upper[2] = upper_pt.z();
      
      int num_cells = patch->getNumExtraCells();
      
      //                                                     Name:                   Example:       
      fwrite( & patch_id,     sizeof(int),    1, fp );    // Patch ID              - 0
      fwrite( & rank_id,      sizeof(int),    1, fp );    // Process ID            - 0
      fwrite(   low_index,    sizeof(int),    3, fp );    // Low Index             - [-1, -1, -1]
      fwrite(   high_index,   sizeof(int),    3, fp );    // High Index            - [20, 20,  4]
      fwrite(   i_low_index,  sizeof(int),    3, fp );    // Interior Low Index    - [ 0,  0,  0]
      fwrite(   i_high_index, sizeof(int),    3, fp );    // Interior High Index   - [20, 20,  3]
      fwrite( & num_nodes,    sizeof(int),    1, fp );    // Number of Extra Nodes - 2646
      fwrite(   lower,        sizeof(double), 3, fp );    // Lower                 - [-0.025, -0.025,  -0.05]
      fwrite(   upper,        sizeof(double), 3, fp );    // Upper                 - [ 0.5, 0.5,  0.2]
      fwrite( & num_cells,    sizeof(int),    1, fp );    // Total number of cells - 2205
    }
  }

  // Write an end of file marker...
  fwrite( &marker,     sizeof(int), 1, fp );

  fclose( fp );

  writeDataTextWriter( hasGlobals, grim_path, grid, procOnLevel );

} // end writeGridBinary()

////////////////////////////////////////////////////////////////
//
// writeGridOriginal()
//
// Creates the <Grid> section of the XML DOM for the output file.
// This original approach places the <Grid> inside the timestep.xml
// file.
//
void
DataArchiver::writeGridOriginal( const bool hasGlobals, const GridP & grid, ProblemSpecP rootElem )
{
  // With AMR, we're not guaranteed that a proc do work on a given
  // level.  Quick check to see that, so we don't create a node that
  // points to no data
  int numLevels = grid->numLevels();
  vector< vector< bool > > procOnLevel( numLevels );

  // Break out the <Grid> and <Data> sections and write those to a
  // "grid.xml" section using libxml2's TextWriter which is a
  // streaming output format which doesn't use a DOM tree.

  ProblemSpecP gridElem = rootElem->appendChild( "Grid" );

  //__________________________________
  //  output level information
  gridElem->appendElement("numLevels", numLevels);

  for(int l=0; l<numLevels; ++l) {
    LevelP level = grid->getLevel(l);
    ProblemSpecP levelElem = gridElem->appendChild("Level");

    if (level->getPeriodicBoundaries() != IntVector(0,0,0)) {
      levelElem->appendElement("periodic", level->getPeriodicBoundaries());
    }
    levelElem->appendElement("numPatches",  level->numPatches());
    levelElem->appendElement("totalCells",  level->totalCells());

    if (level->getExtraCells() != IntVector(0,0,0)) {
      levelElem->appendElement("extraCells", level->getExtraCells());
    }

    levelElem->appendElement("anchor",      level->getAnchor());
    levelElem->appendElement("id",          level->getID());
    levelElem->appendElement("cellspacing", level->dCell());

    //__________________________________
    //  Output patch information:

    procOnLevel[ l ].resize( d_myworld->nRanks() );

    Level::const_patch_iterator iter;
    for( iter = level->patchesBegin(); iter != level->patchesEnd(); ++iter ) {
      const Patch* patch = *iter;
          
      IntVector lo = patch->getCellLowIndex();    // for readability
      IntVector hi = patch->getCellHighIndex();
      IntVector lo_EC = patch->getExtraCellLowIndex();
      IntVector hi_EC = patch->getExtraCellHighIndex();
          
      int proc = m_loadBalancer->getOutputRank( patch );
      procOnLevel[ l ][ proc ] = true;

      Box box = patch->getExtraBox();
      ProblemSpecP patchElem = levelElem->appendChild("Patch");
      
      patchElem->appendElement( "id",        patch->getID() );
      patchElem->appendElement( "proc",      proc );
      patchElem->appendElement( "lowIndex",  patch->getExtraCellLowIndex() );
      patchElem->appendElement( "highIndex", patch->getExtraCellHighIndex() );

      if (patch->getExtraCellLowIndex() != patch->getCellLowIndex()) {
        patchElem->appendElement( "interiorLowIndex", patch->getCellLowIndex() );
      }

      if (patch->getExtraCellHighIndex() != patch->getCellHighIndex()) {
        patchElem->appendElement("interiorHighIndex", patch->getCellHighIndex());
      }

      patchElem->appendElement( "nnodes",     patch->getNumExtraNodes() );
      patchElem->appendElement( "lower",      box.lower() );
      patchElem->appendElement( "upper",      box.upper() );
      patchElem->appendElement( "totalCells", patch->getNumExtraCells() );
    }
  }

  ProblemSpecP dataElem = rootElem->appendChild( "Data" );

  for( int l = 0;l < numLevels; l++ ) {
    ostringstream lname;
    lname << "l" << l;

    // Create a pxxxxx.xml file for each proc doing the outputting.

    for( int i = 0; i < d_myworld->nRanks(); i++ ) {
      if( ( i % m_loadBalancer->getNthRank() ) != 0 || !procOnLevel[l][i] ){
        continue;
      }
          
      ostringstream pname;
      pname << lname.str() << "/p" << setw(5) << setfill('0') << i << ".xml";

      ostringstream procID;
      procID << i;

      ProblemSpecP df = dataElem->appendChild("Datafile");

      df->setAttribute( "href", pname.str() );
      df->setAttribute( "proc", procID.str() );
    }
  }

  if ( hasGlobals ) {
    ProblemSpecP df = dataElem->appendChild( "Datafile" );
    df->setAttribute( "href", "global.xml" );
  }

} // end writeGridOriginal()


void
DataArchiver::writeGridTextWriter( const bool hasGlobals, const string & grim_path, const GridP & grid )
{
  // With AMR, we're not guaranteed that a proc do work on a given
  // level.  Quick check to see that, so we don't create a node that
  // points to no data
  int numLevels = grid->numLevels();
  vector< vector<bool> > procOnLevel( numLevels );

  // Break out the <Grid> and <Data> sections and write those to
  // grid.xml and data.xml files using libxml2's TextWriter which is a
  // streaming output format which doesn't use a DOM tree.

  string name_grid = grim_path + "grid.xml";

  xmlTextWriterPtr writer_grid;
  /* Create a new XmlWriter for uri, with no compression. */
  writer_grid = xmlNewTextWriterFilename( name_grid.c_str(), 0 );
  xmlTextWriterSetIndent( writer_grid, 2 );

# define MY_ENCODING "UTF-8"
  xmlTextWriterStartDocument( writer_grid, nullptr, MY_ENCODING, nullptr );

  xmlTextWriterStartElement( writer_grid, BAD_CAST "Grid" );

  //__________________________________
  //  output level information
  xmlTextWriterWriteFormatElement(writer_grid, BAD_CAST "numLevels","%d", numLevels);

  for (int l=0; l<numLevels; ++l) {
    LevelP level = grid->getLevel(l);

    xmlTextWriterStartElement(writer_grid, BAD_CAST "Level");

    if (level->getPeriodicBoundaries() != IntVector(0,0,0)) {

      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "periodic", "[%d,%d,%d]",
                                       level->getPeriodicBoundaries().x(),
                                       level->getPeriodicBoundaries().y(),
                                       level->getPeriodicBoundaries().z()
                                     );
    }

    xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "numPatches",  "%d", level->numPatches() );
    xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "totalCells", "%ld", level->totalCells() );
    xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "totalCells", "%ld", level->totalCells() );

    if (level->getExtraCells() != IntVector(0,0,0)) {
      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "extraCells", "[%d,%d,%d]",
                                       level->getExtraCells().x(),
                                       level->getExtraCells().y(),
                                       level->getExtraCells().z()
                                     );
    }
    xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "anchor", "[%g,%g,%g]",
                                     level->getAnchor().x(),
                                     level->getAnchor().y(),
                                     level->getAnchor().z()
                                   );
    xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "id", "%d", level->getID() );

    xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "cellspacing", "[%.17g,%.17g,%.17g]",
                                     level->dCell().x(),
                                     level->dCell().y(),
                                     level->dCell().z()
                                   );

    //__________________________________
    //  output patch information
    procOnLevel[ l ].resize( d_myworld->nRanks() );

    Level::const_patch_iterator iter;
    for(iter=level->patchesBegin(); iter != level->patchesEnd(); ++iter) {
      const Patch* patch = *iter;
          
      IntVector lo = patch->getCellLowIndex();    // for readability
      IntVector hi = patch->getCellHighIndex();
      IntVector lo_EC = patch->getExtraCellLowIndex();
      IntVector hi_EC = patch->getExtraCellHighIndex();
          
      int proc = m_loadBalancer->getOutputRank( patch );
      procOnLevel[ l ][ proc ] = true;

      Box box = patch->getExtraBox();

      xmlTextWriterStartElement( writer_grid, BAD_CAST "Patch" );

      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "id",  "%d", patch->getID() );
      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "proc","%d", proc );
      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "lowIndex", "[%d,%d,%d]",
                                       patch->getExtraCellLowIndex().x(),
                                       patch->getExtraCellLowIndex().y(),
                                       patch->getExtraCellLowIndex().z()
                                      );
      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "highIndex", "[%d,%d,%d]",
                                       patch->getExtraCellHighIndex().x(),
                                       patch->getExtraCellHighIndex().y(),
                                       patch->getExtraCellHighIndex().z()
                                      );
      if ( patch->getExtraCellLowIndex() != patch->getCellLowIndex() ){
        xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "interiorLowIndex", "[%d,%d,%d]",
                                         patch->getCellLowIndex().x(),
                                         patch->getCellLowIndex().y(),
                                         patch->getCellLowIndex().z()
                                        );
      }
      if ( patch->getExtraCellHighIndex() != patch->getCellHighIndex() ) {
        xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "interiorHighIndex", "[%d,%d,%d]",
                                         patch->getCellHighIndex().x(),
                                         patch->getCellHighIndex().y(),
                                         patch->getCellHighIndex().z()
                                        );
      }
      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "nnodes", "%d", patch->getNumExtraNodes());
      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "lower",  "[%.17g,%.17g,%.17g]",
                                       box.lower().x(),
                                       box.lower().y(),
                                       box.lower().z()
                                      );
      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "upper", "[%.17g,%.17g,%.17g]",
                                       box.upper().x(),
                                       box.upper().y(),
                                       box.upper().z()
                                      );
      xmlTextWriterWriteFormatElement( writer_grid, BAD_CAST "totalCells", "%d", patch->getNumExtraCells() );
      xmlTextWriterEndElement( writer_grid ); // Close Patch
    }
    xmlTextWriterEndElement( writer_grid ); // Close Level
  }
  xmlTextWriterEndElement(writer_grid); // Close Grid

  xmlTextWriterEndDocument( writer_grid ); // Writes output to the timestep.xml file
  xmlFreeTextWriter( writer_grid );

  writeDataTextWriter( hasGlobals, grim_path, grid, procOnLevel );

} // end writeGridTextWriter()

void
DataArchiver::writeDataTextWriter( const bool hasGlobals, const string & data_path, const GridP & grid,
                                   const vector< vector<bool> > & procOnLevel )
{
  int                   numLevels = grid->numLevels();
  string                data_filename = data_path + "data.xml";
  xmlTextWriterPtr      data_writer = xmlNewTextWriterFilename( data_filename.c_str(), 0 );

  xmlTextWriterSetIndent( data_writer, 2 );
  xmlTextWriterStartElement( data_writer, BAD_CAST "Data" );

  for( int l = 0; l < numLevels; l++ ) {
    ostringstream lname;
    lname << "l" << l;

    // create a pxxxxx.xml file for each proc doing the outputting
    for( int i = 0; i < d_myworld->nRanks(); i++ ) {
      if ( i % m_loadBalancer->getNthRank() != 0 || !procOnLevel[l][i] ) {
        continue;
      }
      ostringstream pname;
      ostringstream procID;

      pname << lname.str() << "/p" << setw(5) << setfill('0') << i << ".xml";
      procID << i;

      xmlTextWriterStartElement( data_writer, BAD_CAST "Datafile" ); // Open <Datafile>

      xmlTextWriterWriteAttribute( data_writer, BAD_CAST "href", BAD_CAST pname.str().c_str() );
      xmlTextWriterWriteAttribute( data_writer, BAD_CAST "proc", BAD_CAST procID.str().c_str() );

      xmlTextWriterEndElement( data_writer ); // Close <Datafile>
    }
  }

  if ( hasGlobals ) {
    xmlTextWriterStartElement( data_writer, BAD_CAST "Datafile" ); // Open <Datafile>
    xmlTextWriterWriteAttribute( data_writer, BAD_CAST "href", BAD_CAST "global.xml" );
    xmlTextWriterEndElement( data_writer ); // Close <Datafile>
  }

  xmlTextWriterEndElement( data_writer );  // Close <Data>
  xmlTextWriterEndDocument( data_writer ); // Writes output to the timestep.xml file
  xmlFreeTextWriter( data_writer );

} // end writeDataTextWriter()

//______________________________________________________________________
//

void
DataArchiver::scheduleOutputTimeStep(       vector<SaveItem> & saveLabels,
                                      const GridP            & grid, 
                                            SchedulerP       & sched,
                                            bool               isThisCheckpoint )
{
  // Schedule a bunch of tasks - one for each variable, for each patch
  int                var_cnt = 0;
  
  for( int i = 0; i < grid->numLevels(); i++ ) {

    const LevelP& level = grid->getLevel(i);
    const PatchSet* patches = m_loadBalancer->getOutputPerProcessorPatchSet( level );
    
    string taskName = "DataArchiver::outputVariables";
    if ( isThisCheckpoint ) {
      taskName += "(checkpoint)";
    }
    
    Task* task = scinew Task( taskName, this, &DataArchiver::outputVariables,
			   isThisCheckpoint ? CHECKPOINT : OUTPUT );
    
    // task->requires( Task::OldDW, m_timeStepLabel);
    
    //__________________________________
    //
    vector< SaveItem >::iterator saveIter;
    for( saveIter = saveLabels.begin();
	 saveIter != saveLabels.end(); ++saveIter ) {
      const MaterialSubset* matls = saveIter->getMaterialSubset(level.get_rep());
      
      if ( matls != nullptr ) {
        task->requires( Task::NewDW, (*saveIter).label, matls, Task::OutOfDomain, Ghost::None, 0, true );
        var_cnt++;
      }
    }

    task->setType( Task::Output );
    sched->addTask( task, patches, m_sharedState->allMaterials() );
  }
  
  if (dbg.active()) {
    dbg << "  scheduled output task for " << var_cnt << " variables\n";
  }
}

#if HAVE_PIDX
void
DataArchiver::createPIDXCommunicator(       vector<SaveItem> & saveLabels,
                                      const GridP            & grid, 
                                            SchedulerP       & sched,
                                            bool               isThisCheckpoint )
{
  int proc = d_myworld->myRank();

  // Resize the comms back to 0...
  m_pidxComms.clear();

  // Create new MPI Comms
  m_pidxComms.reserve( grid->numLevels() );
  
  for( int i = 0; i < grid->numLevels(); i++ ) {

    const LevelP& level = grid->getLevel(i);
    vector< SaveItem >::iterator saveIter;
    const PatchSet* patches = m_loadBalancer->getOutputPerProcessorPatchSet( level );
    //cout << "[ "<< d_myworld->myRank() << " ] Patch size: " << patches->size() << "\n";
    
    /*
      int color = 0;
      if (patches[d_myworld->myRank()].size() != 0)
        color = 1;
      MPI_Comm_split(d_myworld->getComm(), color, d_myworld->myRank(), &(pidxComms[i]));
   */
    
    int color = 0;
    const PatchSubset*  patchsubset = patches->getSubset(proc);
    if (patchsubset->empty() == true) {
      color = 0;
      //cout << "Empty rank: " << proc << "\n";
    }
    else {
      color = 1;
      //cout << "Patch rank: " << proc << "\n";
    }
    
    MPI_Comm_split( d_myworld->getComm(), color, d_myworld->myRank(), &(m_pidxComms[i]) );
    //if (color == 1) {
    //  int nsize;
    //  MPI_Comm_size(pidxComms[i], &nsize);
    //  cout << "NewComm Size = " <<  nsize << "\n";
    //}
  }
}
#endif

//______________________________________________________________________
//
// Be sure to call releaseDocument on the value returned.
ProblemSpecP
DataArchiver::loadDocument(const string & xmlName )
{
  return ProblemSpecReader().readInputFile( xmlName );
}

const string
DataArchiver::getOutputLocation() const
{
    return m_dir.getName();
}

//______________________________________________________________________
//
void
DataArchiver::indexAddGlobals()
{
  if (dbg.active()) {
    dbg << "  indexAddGlobals()\n";
  }

  // add info to index.xml about each global (reduction) var assume
  // for now that global variables that get computed will not change
  // from timestep to timestep
  static bool wereGlobalsAdded = false;
  if (m_writeMeta && !wereGlobalsAdded) {
    wereGlobalsAdded = true;
    // add saved global (reduction) variables to index.xml
    string iname = m_dir.getName()+"/index.xml";
    ProblemSpecP indexDoc = loadDocument(iname);
    
    ProblemSpecP globals = indexDoc->appendChild("globals");

    vector< SaveItem >::iterator saveIter;
    for (saveIter = m_saveReductionLabels.begin();
	 saveIter != m_saveReductionLabels.end(); ++saveIter) {
      SaveItem& saveItem = *saveIter;
      const VarLabel* var = saveItem.label;
      // FIX - multi-level query
      const MaterialSubset* matls = saveItem.getMaterialSet(ALL_LEVELS)->getUnion();
      for (int m = 0; m < matls->size(); m++) {
        int matlIndex = matls->get(m);
        ostringstream href;
        href << var->getName();
        if (matlIndex < 0)
          href << ".dat\0";
        else
          href << "_" << matlIndex << ".dat\0";
        ProblemSpecP newElem = globals->appendChild("variable");
        newElem->setAttribute("href", href.str());
        newElem->setAttribute("type", TranslateVariableType( var->typeDescription()->getName().c_str(), false ) );
        newElem->setAttribute("name", var->getName().c_str());
      }
    }

    indexDoc->output(iname.c_str());
    //indexDoc->releaseDocument();
  }
} // end indexAddGlobals()

//______________________________________________________________________
//
void
DataArchiver::outputReductionVars( const ProcessorGroup *,
                                   const PatchSubset    * /* pss */,
                                   const MaterialSubset * /* matls */,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw )
{
  if( new_dw->timestepRestarted() || m_saveReductionLabels.empty() ) {
    return;
  }

  if (dbg.active()) {
    dbg << "  outputReductionVars task begin\n";
  }

  Timers::Simple timer;
  timer.start();

  double simTime = m_sharedState->getElapsedSimTime();

  // simTime_vartype simTime_var(0);
  // if( old_dw )
  //   old_dw->get( simTime_var, m_simTimeLabel );
  // double simTime = simTime_var;

  delt_vartype delt_var(0);
  if( old_dw )
    old_dw->get( delt_var, m_delTLabel );
  double delT = delt_var;

  // Dump the stuff in the reduction saveset into files in the uda
  // at every timestep
  for(int i=0; i<(int)m_saveReductionLabels.size(); ++i) {
    SaveItem& saveItem = m_saveReductionLabels[i];
    const VarLabel* var = saveItem.label;
    // FIX, see above
    const MaterialSubset* matls =
      saveItem.getMaterialSet(ALL_LEVELS)->getUnion();
    
    for (int m = 0; m < matls->size(); m++) {
      int matlIndex = matls->get(m);

      if (dbg.active()) {
        dbg << "    Reduction " << var->getName() << " matl: " << matlIndex << "\n";
      }

      ostringstream filename;
      filename << m_dir.getName() << "/" << var->getName();
      if (matlIndex < 0)
        filename << ".dat\0";
      else
        filename << "_" << matlIndex << ".dat\0";

#ifdef __GNUG__
      ofstream out(filename.str().c_str(), ios::app);
#else
      ofstream out(filename.str().c_str(), ios_base::app);
#endif
      if (!out) {
        throw ErrnoException("DataArchiver::outputReduction(): The file \"" +
			     filename.str() +
			     "\" could not be opened for writing!",
			     errno, __FILE__, __LINE__);
      }
      
      out << std::setprecision(17) << simTime + delT << "\t";
      new_dw->print(out, var, 0, matlIndex);
      out << "\n";
    }
  }

  double myTime = timer().seconds();
  (*m_runTimeStats)[ReductionIOTime] += myTime;
  (*m_runTimeStats)[TotalIOTime ] += myTime;
  
  dbg << "  outputReductionVars task end\n";
}

//______________________________________________________________________
//
void
DataArchiver::outputVariables( const ProcessorGroup * pg,
                               const PatchSubset    * patches,
                               const MaterialSubset * /*matls*/,
                               DataWarehouse        * old_dw,
                               DataWarehouse        * new_dw,
                               int                    type )
{
  // IMPORTANT - this function should only be called once per
  //   processor per level per type (files will be opened and closed,
  //   and those operations are heavy on parallel file systems)

  // return if not an outpoint/checkpoint timestep
  if ((!m_isOutputTimeStep && type == OUTPUT) || 
      (!m_isCheckpointTimeStep &&
       (type == CHECKPOINT || type == CHECKPOINT_REDUCTION))) {
    return;
  }

  if (dbg.active()) {
    dbg << "  outputVariables task begin\n";
  }

  double timeStep = m_sharedState->getCurrentTopLevelTimeStep();
  
  // timeStep_vartype timeStep_var(0);
  // if( old_dw )
  //   old_dw->get( timeStep_var, m_timeStepLabel );
  // double timeStep = timeStep_var;

#if SCI_ASSERTION_LEVEL >= 2
  // double-check to make sure only called once per level
  int levelid =
    type != CHECKPOINT_REDUCTION ? getLevel(patches)->getIndex() : -1;
  
  if (type == OUTPUT) {
    ASSERT(m_outputCalled[levelid] == false);
    m_outputCalled[levelid] = true;
  }
  else if (type == CHECKPOINT) {
    ASSERT(m_checkpointCalled[levelid] == false);
    m_checkpointCalled[levelid] = true;
  }
  else /* if (type == CHECKPOINT_REDUCTION) */ {
    ASSERT(m_checkpointReductionCalled == false);
    m_checkpointReductionCalled = true;
  }
#endif

  vector< SaveItem >& saveLabels =
	 (type == OUTPUT ? m_saveLabels :
	  type == CHECKPOINT ? m_checkpointLabels : m_checkpointReductionLabels);

  //__________________________________
  // debugging output
  // this task should be called once per variable (per patch/matl subset).
  if (dbg.active()) {
    if (type == CHECKPOINT_REDUCTION) {
      dbg << "    reduction";
    }
    else /* if (type == OUTPUT || type == CHECKPOINT) */ {
      if (type == CHECKPOINT)
        dbg << "    checkpoint ";
        
      dbg << "    patches: ";
      for(int p=0;p<patches->size();p++) {
        if(p != 0)
          dbg << ", ";
        if (patches->get(p) == 0)
          dbg << -1;
        else
          dbg << patches->get(p)->getID();
      }
    }
    
    dbg << " on timestep: " << timeStep << "\n";
  }
    
  
  //__________________________________
  Dir dir;
  if (type == OUTPUT) {
    dir = m_dir;
  }
  else /* if (type == CHECKPOINT || type == CHECKPOINT_REDUCTION) */ {
    dir = m_checkpointsDir;
  }
  
  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << getTimeStepTopLevel();

  Dir tdir = dir.getSubdir( tname.str() );
  Dir ldir;
  
  string xmlFilename;
  string dataFilebase;
  string dataFilename;
  const Level* level = nullptr;

  // find the xml filename and data filename that we will write to
  // Normal reductions will be handled by outputReduction, but
  // checkpoint reductions call this function, and we handle them
  // differently.
  if (type == OUTPUT || type == CHECKPOINT) {
    // find the level and level number associated with this patch
    
    ASSERT(patches->size() != 0);
    ASSERT(patches->get(0) != 0);
    level = patches->get(0)->getLevel();
    
#if SCI_ASSERTION_LEVEL >= 1
    for(int i=0;i<patches->size();i++){
      ASSERT(patches->get(i)->getLevel() == level);
    }
#endif

    ostringstream lname;
    lname << "l" << level->getIndex();
    ldir = tdir.getSubdir(lname.str());
    
    ostringstream pname;
    pname << "p" << setw(5) << setfill('0') << d_myworld->myRank();
    xmlFilename = ldir.getName() + "/" + pname.str() + ".xml";
    dataFilebase = pname.str() + ".data";
    dataFilename = ldir.getName() + "/" + dataFilebase;
  }
  else /* if (type == CHECKPOINT_REDUCTION) */ {
    xmlFilename =  tdir.getName() + "/global.xml";
    dataFilebase = "global.data";
    dataFilename = tdir.getName() + "/" + dataFilebase;
  }

  Timers::Simple timer;
  timer.start();
  
  size_t totalBytes = 0;  // total bytes saved over all variables

  //__________________________________
  // Output using standard output format
  // Not only lock to prevent multiple threads from writing over the same
  // file, but also lock because xerces (DOM..) has thread-safety issues.

  if (m_outputFileFormat == UDA || type == CHECKPOINT_REDUCTION)
  {  
    m_outputLock.lock(); 
    {  

#if 0
      // DON'T reload a timestep.xml - it will probably mean there was
      // a timestep restart that had written data and we will want to
      // overwrite it
      ifstream test(xmlFilename.c_str());
      if(test) {
        doc = loadDocument(xmlFilename);
      } else
#endif
      // make sure doc's constructor is called after the lock.
      ProblemSpecP doc = ProblemSpec::createDocument("Uintah_Output");
      // Find the end of the file
      ASSERT(doc != nullptr);
      ProblemSpecP n = doc->findBlock("Variable");
      
      long cur=0;
      while(n != nullptr) {
        ProblemSpecP endNode = n->findBlock("end");
        ASSERT(endNode != nullptr);
        long end = atol(endNode->getNodeValue().c_str());
        
        if(end > cur)
          cur=end;
        n = n->findNextBlock("Variable");
      }
      
      //__________________________________
      // Open the data file:
      //
      // Note: At least one time on a BGQ machine (Vulcan@LLNL), with
      // 160K patches, a single checkpoint file failed to open, and it
      // 'crashed' the simulation.  As the other processes on the node
      // successfully opened their file, it is possible that a second
      // open call would have succeeded.  (The original error no was
      // 71.)  Therefore I am using a while loop and counting the
      // 'tries'.
      
      int tries = 1;
      int flags = O_WRONLY|O_CREAT|O_TRUNC;       // file-opening flags
      
      const char* filename = dataFilename.c_str();
      int fd  = open( filename, flags, 0666 );
      
      while( fd == -1 ) {

        if( tries >= 50 ) {
          ostringstream msg;
          
          msg << "DataArchiver::output(): Failed to open file '"
	      << dataFilename << "' (after 50 tries).";
          throw ErrnoException( msg.str(), errno, __FILE__, __LINE__ );
        }

        fd = open( filename, flags, 0666 );
        tries++;
      }
      
      if( tries > 1 ) {
        proc0cout << "WARNING: There was a glitch in trying to open the "
		  << "checkpoint file: " << dataFilename << ". "
		  << "It took " << tries << " tries to successfully open it.";
      }

      //__________________________________
      // loop over variables
      vector< SaveItem >::iterator saveIter;
      for(saveIter = saveLabels.begin();
	  saveIter != saveLabels.end(); ++saveIter) {	
        const VarLabel* var = saveIter->label;
        
        const MaterialSubset* var_matls = saveIter->getMaterialSubset(level);
        
        if (var_matls == 0)
          continue;
        
        //__________________________________
        //  debugging output
        if (dbg.active()) {
          dbg << "    " << var->getName() << ", materials: ";
          for (int m = 0; m < var_matls->size(); m++) {
            if (m != 0)
              dbg << ", ";
            dbg << var_matls->get(m);
          }
          dbg << "\n";
        }
        
        //__________________________________
        // loop through patches and materials
        for(int p=0; p<(type==CHECKPOINT_REDUCTION?1:patches->size()); ++p) {
          const Patch* patch;
          int patchID;
          
          if (type == CHECKPOINT_REDUCTION) {
            // to consolidate into this function, force patch = 0
            patch = 0;
            patchID = -1;
          }
	  else /* if (type == OUTPUT || type == CHECKPOINT) */ {
            patch = patches->get(p);
            patchID = patch->getID();
          }
          
          //__________________________________
          // write info for this variable to current index file
          for(int m=0;m<var_matls->size();m++) {
            
            int matlIndex = var_matls->get(m);
            
            // Variables may not exist when we get here due to
            // something whacky with weird AMR stuff...
            ProblemSpecP pdElem = doc->appendChild("Variable");
            
            pdElem->appendElement("variable", var->getName());
            pdElem->appendElement("index", matlIndex);
            pdElem->appendElement("patch", patchID);
            pdElem->setAttribute("type",TranslateVariableType( var->typeDescription()->getName().c_str(), type != OUTPUT ) );
            
            if (var->getBoundaryLayer() != IntVector(0,0,0)) {
              pdElem->appendElement("boundaryLayer", var->getBoundaryLayer());
            }
#if 0          
            off_t ls = lseek(fd, cur, SEEK_SET);
            
            if(ls == -1) {
              cerr << "lseek error - file: " << filename
		   << ", errno=" << errno << '\n';
              throw ErrnoException("DataArchiver::output (lseek call)",
				   errno, __FILE__, __LINE__);
            }
#endif
            // Pad appropriately
            if(cur%PADSIZE != 0) {
              long pad = PADSIZE-cur%PADSIZE;
              char* zero = scinew char[pad];
              memset(zero, 0, pad);
              int err = (int)write(fd, zero, pad);
              if (err != pad) {
                cerr << "Error writing to file: " << filename
		     << ", errno=" << errno << '\n';
                SCI_THROW(ErrnoException("DataArchiver::output (write call)",
					 errno, __FILE__, __LINE__));
              }
              cur+=pad;
              delete[] zero;
            }
            ASSERTEQ(cur%PADSIZE, 0);
            pdElem->appendElement("start", cur);
            
            // output data to data file
            OutputContext oc(fd, filename, cur, pdElem, m_outputDoubleAsFloat && type != CHECKPOINT);
            totalBytes += new_dw->emit(oc, var, matlIndex, patch);
            
            pdElem->appendElement("end", oc.cur);
            pdElem->appendElement("filename", dataFilebase.c_str());
            
#if SCI_ASSERTION_LEVEL >= 1
            struct stat st;
            int s = fstat(fd, &st);
            
            if(s == -1) {
              cerr << "fstat error - file: " << filename
		   << ", errno=" << errno << '\n';
              throw ErrnoException("DataArchiver::output (stat call)",
				   errno, __FILE__, __LINE__);
            }
            ASSERTEQ(oc.cur, st.st_size);
#endif
            
            cur=oc.cur;
          }  // matls
        }  // patches
      }  // save items
      
      //__________________________________
      // close files and handles 
      int s = close(fd);
      if(s == -1) {
        cerr << "Error closing file: " << filename
	     << ", errno=" << errno << '\n';
        throw ErrnoException("DataArchiver::output (close call)",
			     errno, __FILE__, __LINE__);
      }
      
      doc->output(xmlFilename.c_str());
      //doc->releaseDocument();
    }

    m_outputLock.unlock(); 
  }

#if HAVE_PIDX
  //______________________________________________________________________
  //
  //  ToDo
  //      Multiple patches per core. Needed for outputNthProc (Sidharth)
  //      Turn off output from inside of PIDX (Sidharth)
  //      Disable need for MPI in PIDX (Sidharth)
  //      Fix ints issue in PIDX (Sidharth)
  //      Do we need the memset calls?
  //      Is Variable::emitPIDX() and Variable::readPIDX() efficient? 
  //      Should we be using calloc() instead of malloc+memset
  //
  if ( m_outputFileFormat == PIDX && type != CHECKPOINT_REDUCTION ) {
  
    //__________________________________
    // create the xml dom for this variable
    ProblemSpecP doc = ProblemSpec::createDocument("Uintah_Output-PIDX");
    ASSERT(doc != nullptr);
    ProblemSpecP n = doc->findBlock("Variable");
    while( n != nullptr ) {
      n = n->findNextBlock("Variable");
    }  
      
    PIDXOutputContext pidx;
    vector<TypeDescription::Type> GridVarTypes =
      pidx.getSupportedVariableTypes();
    
    //bulletproofing
    isVarTypeSupported( saveLabels, GridVarTypes );
    
    // loop over the grid variable types.
    vector<TypeDescription::Type>::iterator iter;
    for(iter = GridVarTypes.begin(); iter!= GridVarTypes.end(); ++iter) {
      TypeDescription::Type TD = *iter;
      
      // find all variables of this type
      vector<SaveItem> saveTheseLabels = findAllVariableTypes( saveLabels, TD );
      
      if( saveTheseLabels.size() > 0 ) {
        string dirName = pidx.getDirectoryName( TD );

        Dir myDir = ldir.getSubdir( dirName );
        
        totalBytes += saveLabels_PIDX( pg, patches, new_dw, type,
				       saveTheseLabels, TD, ldir, dirName, doc);
      } 
    }

    // write the xml 
    //doc->output(xmlFilename.c_str());
  }
#endif

  double myTime = timer().seconds();
  double byteToMB = 1024*1024;

  if (type == OUTPUT) {
    (*m_runTimeStats)[OutputIOTime] +=
      myTime;
    (*m_runTimeStats)[OutputIORate] +=
      (double) totalBytes / (byteToMB * myTime);
  }
  else if (type == CHECKPOINT ) {
    (*m_runTimeStats)[CheckpointIOTime] +=
      myTime;
    (*m_runTimeStats)[CheckpointIORate] +=
      (double) totalBytes / (byteToMB * myTime);
  }
    
  else /* if (type == CHECKPOINT_REDUCTION) */ {
    (*m_runTimeStats)[CheckpointReductionIOTime] +=
      myTime;
    (*m_runTimeStats)[CheckpointReducIORate] +=
      (double) totalBytes / (byteToMB * myTime);
  }
    
  (*m_runTimeStats)[TotalIOTime ] += myTime;

  if (dbg.active()) {
    dbg << "  outputVariables task end\n";
  }
} // end outputVariables()

//______________________________________________________________________
//  output only the savedLabels of a specified type description in PIDX format.

size_t
DataArchiver::saveLabels_PIDX( const ProcessorGroup        * pg,
                               const PatchSubset           * patches,      
                                     DataWarehouse         * new_dw,          
                               int                           type,
                               std::vector< SaveItem >     & saveLabels,
                               const TypeDescription::Type   TD,
                               Dir                           ldir,        // uda/timestep/levelIndex
                               const std::string           & dirName,     // CCVars, SFC*Vars
                               ProblemSpecP                & doc )
{
  size_t totalBytesSaved = 0;
#if HAVE_PIDX
  const int timeStep = m_application->getTimeStep();

  int levelid = getLevel(patches)->getIndex(); 
  const Level* level = getLevel(patches);

  int nSaveItems =  saveLabels.size();
  vector<int> nSaveItemMatls (nSaveItems);

  int rank = pg->myRank();
  int rc = -9;               // PIDX return code

  //__________________________________
  // Count up the number of variables that will 
  // be output. Each variable can have a different number of 
  // materials and live on a different number of levels
  
  int count = 0;
  int actual_number_of_variables = 0;
  
  vector< SaveItem >::iterator saveIter;
  for (saveIter = saveLabels.begin(); saveIter != saveLabels.end(); ++saveIter) {
    const MaterialSubset* var_matls = saveIter->getMaterialSubset(level);

    if (var_matls == nullptr) {
      continue;
    }

    nSaveItemMatls[count] = var_matls->size();

    ++count;
    actual_number_of_variables += var_matls->size();
  }

  // must use this format or else file won't be written
  // inside of uda
  Dir myDir = ldir.getSubdir( dirName );                   // uda/timestep/level/<CCVars, SFC*Vars>
  string idxFilename ( dirName + ".idx" );                 // <CCVars, SFC*Vars....>.idx
  string full_idxFilename( myDir.getName() + ".idx" );     // uda/timestep/level/<CCVars, SFC*Vars...>.idx
  
  PIDXOutputContext pidx;

  pidx.setOutputDoubleAsFloat( (m_outputDoubleAsFloat && type == OUTPUT) );  

  //__________________________________
  // define the level extents for this variable type
  IntVector lo;
  IntVector hi;
  level->computeVariableExtents(TD,lo, hi);
  
  PIDX_point level_size;
  pidx.setLevelExtents( "DataArchiver::saveLabels_PIDX",  lo, hi, level_size );

  // Can this be run in serial without doing a MPI initialize
  pidx.initialize( full_idxFilename, timeStep, /*d_myworld->getComm()*/m_pidxComms[ levelid ], m_PIDX_flags, patches, level_size, type );

  //__________________________________
  // allocate memory for pidx variable descriptor array
  rc = PIDX_set_variable_count(pidx.file, actual_number_of_variables);
  pidx.checkReturnCode( rc, "DataArchiver::saveLabels_PIDX -PIDX_set_variable_count failure",__FILE__, __LINE__);
  
  size_t pidxVarSize = sizeof (PIDX_variable*);
  pidx.varDesc = (PIDX_variable**) malloc( pidxVarSize * nSaveItems );
  memset(pidx.varDesc, 0, pidxVarSize * nSaveItems);

  for(int i = 0 ; i < nSaveItems ; i++) {
    pidx.varDesc[i] = (PIDX_variable*) malloc( pidxVarSize * nSaveItemMatls[i]);
    memset(pidx.varDesc[i], 0, pidxVarSize * nSaveItemMatls[i]);
  }

  //__________________________________
  //  PIDX Diagnostics
  if( rank == 0 && dbgPIDX.active()) {
    printf("[PIDX] IDX file name = %s\n", (char*)idxFilename.c_str());
    printf("[PIDX] levelExtents = %d %d %d\n", (hi.x() - lo.x()), (hi.y() - lo.y()), (hi.z() - lo.z()) );
    printf("[PIDX] Total number of variable = %d\n", nSaveItems);
  }

  //__________________________________
  //
  unsigned char ***patch_buffer;
  patch_buffer = (unsigned char***)malloc(sizeof(unsigned char**) * actual_number_of_variables);
  
  int vc = 0;
  int vcm = 0;
  
  for(saveIter = saveLabels.begin(); saveIter != saveLabels.end(); saveIter++) 
  {
    const VarLabel* label = saveIter->label;

    const MaterialSubset* var_matls = saveIter->getMaterialSubset(level);
    if (var_matls == nullptr){
      continue;
    }

    //__________________________________
    //  determine data subtype sizes
    const TypeDescription* td = label->typeDescription();
    const TypeDescription* subtype = td->getSubType();      

    // set values depending on the variable's subtype
    char data_type[512];
    int sample_per_variable = -9;
    size_t varSubType_size = -9;
    
    switch( subtype->getType( )) {

      case Uintah::TypeDescription::Stencil7:
        sample_per_variable = 7;
        varSubType_size = sample_per_variable * sizeof(double);
        sprintf(data_type, "%d*float64", sample_per_variable); 
        break;
      case Uintah::TypeDescription::Stencil4:
        sample_per_variable = 4;
        varSubType_size = sample_per_variable * sizeof(double);
        sprintf(data_type, "%d*float64", sample_per_variable); 
        break;
        
      case Uintah::TypeDescription::Vector:
        sample_per_variable = 3;
        varSubType_size = sample_per_variable * sizeof(double);
        sprintf(data_type, "%d*float64", sample_per_variable);
        break;

      case Uintah::TypeDescription::int_type:
        sample_per_variable = 1;
        varSubType_size = sample_per_variable * sizeof(int);
        sprintf(data_type, "%d*int32", sample_per_variable);
        break;
        
      case Uintah::TypeDescription::double_type:
        sample_per_variable = 1;
        
        // take into account saving doubles as floats
        if ( pidx.isOutputDoubleAsFloat() ){
          varSubType_size = sample_per_variable * sizeof(float);
          sprintf(data_type, "%d*float32", sample_per_variable);
        } else {
          varSubType_size = sample_per_variable * sizeof(double);
          sprintf(data_type, "%d*float64", sample_per_variable);
        }
        break;
      default:
        ostringstream warn;
        warn << "DataArchiver::saveLabels_PIDX:: ("<< label->getName() << " " 
             << td->getName() << " ) has not been implemented\n";
        throw InternalError(warn.str(), __FILE__, __LINE__); 
    }

    //__________________________________
    //  materials loop
    for(int m=0;m<var_matls->size();m++){
      int matlIndex = var_matls->get(m);
      string var_mat_name;

      if (var_matls->size() == 1) {
        var_mat_name = label->getName();
      } else {
        std::ostringstream s;
        s << m;
        var_mat_name = label->getName() + "_m" + s.str();
      }
    
      rc = PIDX_variable_create((char*) var_mat_name.c_str(),
				varSubType_size * 8, data_type,
				&(pidx.varDesc[vc][m]));
      
      pidx.checkReturnCode( rc,
			    "DataArchiver::saveLabels_PIDX - PIDX_variable_create failure",
			    __FILE__, __LINE__);
      

      patch_buffer[vcm] =
	(unsigned char**) malloc(sizeof(unsigned char*) * patches->size());

      //__________________________________
      //  patch Loop
      for( int p=0; p<(type == CHECKPOINT_REDUCTION ? 1 : patches->size()); ++p ) {
        const Patch* patch;

        if (type == CHECKPOINT_REDUCTION) {
          patch = 0;
        }
        else {
          patch = patches->get(p);      
          PIDX_point patchOffset;
          PIDX_point patchSize;
          PIDXOutputContext::patchExtents patchExts;
          
          pidx.setPatchExtents( "DataArchiver::saveLabels_PIDX",
				patch, level, label->getBoundaryLayer(),
                                td, patchExts, patchOffset, patchSize );
                              
          //__________________________________
          // debugging
          if( dbgPIDX.active() && isProc0_macro ) {
            proc0cout << rank <<" taskType: " << type << "  PIDX:  "
		      << setw(15) << label->getName() << "  " << td->getName() 
                      << " Patch: " << patch->getID()
		      << " L-" << level->getIndex() 
                      << ",  sample_per_variable: " << sample_per_variable
		      << " varSubType_size: " << varSubType_size
		      << " dataType: " << data_type << "\n";
            patchExts.print(cout); 
          }
                    
          //__________________________________
          // allocate memory for the grid variables
          size_t arraySize = varSubType_size * patchExts.totalCells_EC;

          patch_buffer[vcm][p] = (unsigned char*)malloc( arraySize );
          memset( patch_buffer[vcm][p], 0, arraySize );
          
          //__________________________________
          //  Read in Array3 data to t-buffer
          new_dw->emitPIDX(pidx, label, matlIndex, patch, patch_buffer[vcm][p], arraySize);

          #if 0           // to hardwire buffer values for debugging.
          pidx.hardWireBufferValues(t_buffer, patchExts, arraySize, sample_per_variable );
          #endif
          

          //__________________________________
          //  debugging
          if (dbgPIDX.active() ){
             pidx.printBufferWrap("DataArchiver::saveLabels_PIDX    BEFORE  PIDX_variable_write_data_layout",
                                   subtype->getType(),
                                   sample_per_variable,        
                                   patchExts.lo_EC, patchExts.hi_EC,
                                   patch_buffer[vcm][p],                          
                                   arraySize );
          }         
         
          rc = PIDX_variable_write_data_layout(pidx.varDesc[vc][m],
					       patchOffset, patchSize,
					       patch_buffer[vcm][p],
					       PIDX_row_major);
	  
          pidx.checkReturnCode( rc,
				"DataArchiver::saveLabels_PIDX - PIDX_variable_write_data_layout failure",
				__FILE__, __LINE__);
          
          totalBytesSaved += arraySize;
          
          //__________________________________	  
          //  populate the xml dom This layout allows us to highjack
          //  all of the existing data structures in DataArchive
          ProblemSpecP var_ps = doc->appendChild("Variable");
          var_ps->setAttribute( "type",      TranslateVariableType( td->getName().c_str(), type != OUTPUT ) );
          var_ps->appendElement("variable",  label->getName());
          var_ps->appendElement("index",     matlIndex);
          var_ps->appendElement("patch",     patch->getID());                     
          var_ps->appendElement("start",     vcm);                         
          var_ps->appendElement("end",       vcm);          // redundant
          var_ps->appendElement("filename",  idxFilename);

          if (label->getBoundaryLayer() != IntVector(0,0,0)) {
            var_ps->appendElement("boundaryLayer", label->getBoundaryLayer());
          }
        }  // is checkpoint?
      }  //  Patches

      rc = PIDX_appenm_anm_write_variable(pidx.file, pidx.varDesc[vc][m]);
      pidx.checkReturnCode( rc,
			    "DataArchiver::saveLabels_PIDX - PIDX_appenm_anm_write_variable failure",
			    __FILE__, __LINE__);
      
      vcm++;
    }  //  Materials

    vc++;
  }  //  Variables

  rc = PIDX_close(pidx.file);
  pidx.checkReturnCode( rc,
			"DataArchiver::saveLabels_PIDX - PIDX_close failure",
			__FILE__, __LINE__);
  
  //__________________________________
  // free buffers
  for (int i=0;i<actual_number_of_variables;i++)
  {
    for(int p=0;p<(type==CHECKPOINT_REDUCTION?1:patches->size());p++)
    {
      free( patch_buffer[i][p] );
      patch_buffer[i][p] = 0;
    }
    free( patch_buffer[i] );
    patch_buffer[i] = 0;
  }

  free(patch_buffer);
  patch_buffer = 0;

  //__________________________________
  //  free memory
  for (int i=0; i<nSaveItems ; i++){
    free(pidx.varDesc[i]);
  }

  free(pidx.varDesc); 
  pidx.varDesc=0;

#endif

//cout << "   totalBytesSaved: " << totalBytesSaved << " nSavedItems: " << nSaveItems << "\n";
return totalBytesSaved;
}

//______________________________________________________________________
//  Return a vector of saveItems with a common typeDescription
std::vector<DataArchiver::SaveItem> 
DataArchiver::findAllVariableTypes( std::vector< SaveItem >& saveLabels,
                                    const TypeDescription::Type TD )
{
  std::vector< SaveItem > myItems;
  myItems.empty();

  vector< SaveItem >::iterator saveIter;  
  for (saveIter = saveLabels.begin(); saveIter != saveLabels.end(); ++saveIter) {
    const VarLabel* label = saveIter->label;
    TypeDescription::Type myType = label->typeDescription()->getType();
    
    if( myType == TD){
      myItems.push_back( *saveIter );
    }
  }
  return myItems;
}


//______________________________________________________________________
//  throw exception if saveItems type description is NOT supported 
void 
DataArchiver::isVarTypeSupported( std::vector< SaveItem >& saveLabels,
                                  std::vector<TypeDescription::Type> pidxVarTypes )
{ 
  vector< SaveItem >::iterator saveIter;  
  for (saveIter = saveLabels.begin(); saveIter != saveLabels.end(); ++saveIter) {
    const VarLabel* label = saveIter->label;
    const TypeDescription* myType = label->typeDescription();
    
    bool found = false;
    vector<TypeDescription::Type>::iterator tm_iter;
    for (tm_iter = pidxVarTypes.begin(); tm_iter!= pidxVarTypes.end(); ++tm_iter) {
      TypeDescription::Type TD = *tm_iter;
      if( myType->getType() == TD ){
        found = true;
        continue;
      }
    }
    
    // throw exception if this type isn't supported
    if( found == false){
      ostringstream warn;
      warn << "DataArchiver::saveLabels_PIDX:: ("<< label->getName() << ",  " 
           << myType->getName() << " ) has not been implemented";
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
  }
}

//______________________________________________________________________
//  Create the PIDX sub directories
void
DataArchiver::createPIDX_dirs( std::vector< SaveItem >& saveLabels,
                               Dir& levelDir )
{
#if HAVE_PIDX

  if( m_outputFileFormat == UDA ) {
    return;
  }
  
  PIDXOutputContext pidx;
  vector<TypeDescription::Type> GridVarTypes = pidx.getSupportedVariableTypes();

  // loop over the grid variable types.
  vector<TypeDescription::Type>::iterator iter;
  for(iter = GridVarTypes.begin(); iter!= GridVarTypes.end(); ++iter) {
    TypeDescription::Type TD = *iter;

    // find all variables of this type
    vector<SaveItem> saveTheseLabels;
    saveTheseLabels = findAllVariableTypes( saveLabels, TD );

    // create the sub directories
    if( saveTheseLabels.size() > 0 ) {
      string dirName = pidx.getDirectoryName(TD);
      Dir myDir = levelDir.createSubdirPlus( dirName );
    }
  }
#endif
}

//______________________________________________________________________
//
void
DataArchiver::makeVersionedDir()
{
  unsigned int dirMin = 0;
  unsigned int dirNum = 0;
  unsigned int dirMax = 0;

  bool dirCreated = false;

  // My (Dd) understanding of this while loop is as follows: We try to
  // make a new directory starting at 0 (000), then 1 (001), and then
  // doubling the number each time that the directory already exists.
  // Once we find a directory number that does not exist, we start
  // back at the minimum number that we had already tried and work
  // forward to find an actual directory number that we can use.
  //
  // Eg: If 001 and 002 already exist, then the algorithm tries to
  // create 001, fails (it already exists), tries to created 002,
  // fails again, tries to create 004, succeeds, deletes 004 because
  // we aren't sure that it is the smallest new number, creates 003,
  // succeeds, and since 003 is the smallest (dirMin) the algorithm
  // stops.
  //
  // This routine has been re-written to not use the Dir class because
  // the Dir class throws exceptions and exceptions (under SGI CC) add
  // a huge memory penalty if one (same penalty if more than) is
  // thrown.  This causes memory usage to change if the very first
  // directory (000) can be created (because no exception is thrown
  // and thus no memory is allocated for exceptions).
  //
  // If there is a real error, then we can throw an exception because
  // we don't care about the memory penalty.

  string dirName;

  // first check to see if the suffix passed in on the command line
  // makes a valid uda dir

  if (m_udaSuffix != -1) {
    ostringstream name;
    name << m_filebase << "." << setw(3) << setfill('0') << m_udaSuffix;
    dirName = name.str();
    
    int code = MKDIR( dirName.c_str(), 0777 );
    if( code == 0 ) { // Created the directory successfully
      dirCreated = true;
    }
    else if( errno != EEXIST )  {
      cerr << "makeVersionedDir: Error making directory: " << name.str()
	   << "\n";
      throw ErrnoException("DataArchiver.cc: mkdir failed for some "
                           "reason besides dir already exists", errno,
			   __FILE__, __LINE__);
    }
  }

  // If that didn't work, go ahead with the real algorithm

  while (!dirCreated) {
    ostringstream name;
    name << m_filebase << "." << setw(3) << setfill('0') << dirNum;
    dirName = name.str();
      
    int code = MKDIR( dirName.c_str(), 0777 );
    if( code == 0 ) {// Created the directory successfully
      dirMax = dirNum;
      if (dirMax == dirMin)
        dirCreated = true;
      else {
        int code = rmdir( dirName.c_str() );
        if (code != 0)
          throw ErrnoException("DataArchiver.cc: rmdir failed", errno,
			       __FILE__, __LINE__);
      }
    }
    else {
      if( errno != EEXIST ) {
        cerr << "makeVersionedDir: Error making directory: " << name.str()
	     << "\n";
        throw ErrnoException("DataArchiver.cc: mkdir failed for some "
                             "reason besides dir already exists", errno,
			     __FILE__, __LINE__);
      }
      dirMin = dirNum + 1;
    }

    if (!dirCreated) {
      if (dirMax == 0) {
        if (dirNum == 0)
          dirNum = 1;
        else
          dirNum *= 2;
      } else {
        dirNum = dirMin + ((dirMax - dirMin) / 2);
      }
    }
  }
  // Move the symbolic link to point to the new version. We need to be careful
  // not to destroy data, so we only create the link if no file/dir by that
  // name existed or if it's already a link.
  bool make_link = false;
  struct stat sb;
  int rc = LSTAT(m_filebase.c_str(), &sb);
  if ((rc != 0) && (errno == ENOENT))
    make_link = true;
  else if ((rc == 0) && (S_ISLNK(sb.st_mode))) {
    unlink(m_filebase.c_str());
    make_link = true;
  }
  if (make_link)
    symlink(dirName.c_str(), m_filebase.c_str());

  cout << "DataArchiver created " << dirName << "\n";
  m_dir = Dir(dirName);
   
} // end makeVersionedDir()


//______________________________________________________________________
//  Determine which labels will be saved.
void
DataArchiver::initSaveLabels(SchedulerP& sched, bool initTimeStep)
{
  if (dbg.active()) {
    dbg << "  initSaveLabels()\n";
  }

  // if this is the initTimeStep, then don't complain about saving all
  // the vars, just save the ones you can.  They'll most likely be
  // around on the next timestep.
 
  SaveItem saveItem;
  m_saveReductionLabels.clear();
  m_saveLabels.clear();
   
  m_saveLabels.reserve( m_saveLabelNames.size() );
  Scheduler::VarLabelMaterialMap* pLabelMatlMap;
  pLabelMatlMap = sched->makeVarLabelMaterialMap();

  // iterate through each of the saveLabelNames we created in problemSetup
  list<SaveNameItem>::iterator iter;
  for (iter = m_saveLabelNames.begin();
       iter != m_saveLabelNames.end(); ++iter) {
    VarLabel* var = VarLabel::find((*iter).labelName);
    
    //   see if that variable has been created, set the compression
    //   mode make sure that the scheduler shows that that it has been
    //   scheduled to be computed.  Then save it to saveItems.
    if (var == nullptr) {
      if (initTimeStep) {
        continue;
      } else {
        throw ProblemSetupException((*iter).labelName +" variable not found to save.", __FILE__, __LINE__);
      }
    }
    
    if ((*iter).compressionMode != "") {
      var->setCompressionMode((*iter).compressionMode);
    }
      
    Scheduler::VarLabelMaterialMap::iterator found =
      pLabelMatlMap->find(var->getName());

    if (found == pLabelMatlMap->end()) {
      if (initTimeStep) {
        // ignore this on the init timestep, cuz lots of vars aren't
        // computed on the init timestep

        if (dbg.active()) {
          dbg << "    Ignoring var " << iter->labelName << " on initialization timestep\n";
        }

        continue;
      }
      else {
        throw ProblemSetupException((*iter).labelName + " variable not computed for saving.", __FILE__, __LINE__);
      }
    }
    saveItem.label = var;
    saveItem.matlSet.clear();

    for (ConsecutiveRangeSet::iterator crs_iter = (*iter).levels.begin(); crs_iter != (*iter).levels.end(); ++crs_iter) {

      ConsecutiveRangeSet matlsToSave = (ConsecutiveRangeSet((*found).second)).intersected((*iter).matls);
      saveItem.setMaterials(*crs_iter, matlsToSave, m_prevMatls, m_prevMatlSet);

      if (((*iter).matls != ConsecutiveRangeSet::all) && ((*iter).matls != matlsToSave)) {
        throw ProblemSetupException((*iter).labelName + " variable not computed for all materials specified to save.",
        __FILE__,
                                    __LINE__);
      }
    }

    if (saveItem.label->typeDescription()->isReductionVariable()) {
      m_saveReductionLabels.push_back(saveItem);
    }
    else {
      m_saveLabels.push_back(saveItem);
    }
  }
  
  //m_saveLabelNames.clear();
  delete pLabelMatlMap;
}


//______________________________________________________________________
//
void
DataArchiver::initCheckpoints(SchedulerP& sched)
{
  if (dbg.active()) {
    dbg << "  initCheckpoints()\n";
  }

   typedef vector<const Task::Dependency*> dep_vector;
   const dep_vector& initreqs = sched->getInitialRequires();
   
   // special variables to not checkpoint
   const set<string>& notCheckPointVars = sched->getNotCheckPointVars();
   
   SaveItem saveItem;
   m_checkpointReductionLabels.clear();
   m_checkpointLabels.clear();

   // label -> level -> matls
   // we can't store them in two different maps, since we need to know 
   // which levels depend on which materials
   typedef map<string, map<int, ConsecutiveRangeSet> > label_type;
   label_type label_map;

   dep_vector::const_iterator iter;
   for (iter = initreqs.begin(); iter != initreqs.end(); ++iter) {
     const Task::Dependency* dep = *iter;

     // define the patchset
     const PatchSubset* patchSubset = (dep->m_patches != 0)? dep->m_patches : dep->m_task->getPatchSet()->getUnion();
     
     // adjust the patchSubset if the dependency requires coarse or fine level patches
     constHandle<PatchSubset> patches;     
     if ( dep->m_patches_dom == Task::CoarseLevel || dep->m_patches_dom == Task::FineLevel ){
       patches = dep->getPatchesUnderDomain( patchSubset );
       patchSubset = patches.get_rep();
     }
   
     // Define the Levels
     ConsecutiveRangeSet levels;
     for(int i=0;i<patchSubset->size();i++) {
       const Patch* patch = patchSubset->get(i); 
       levels.addInOrder(patch->getLevel()->getIndex());
     }

     //  MaterialSubset:
     const MaterialSubset* matSubset = (dep->m_matls != 0) ? dep->m_matls : dep->m_task->getMaterialSet()->getUnion();
     
     // The matSubset is assumed to be in ascending order or
     // addInOrder will throw an exception.
     ConsecutiveRangeSet matls;
     matls.addInOrder(matSubset->getVector().begin(), matSubset->getVector().end());

     for(ConsecutiveRangeSet::iterator crs_iter = levels.begin(); crs_iter != levels.end(); ++crs_iter) {
       ConsecutiveRangeSet& unionedVarMatls = label_map[dep->m_var->getName()][*crs_iter];
       unionedVarMatls = unionedVarMatls.unioned(matls);
     }
     //cout << "  Adding checkpoint var " << dep->m_var->getName() << " levels " << levels << " matls " << matls << "\n";
   }
         
   m_checkpointLabels.reserve(label_map.size());
   bool hasDelT = false;

   label_type::iterator lt_iter;
   for (lt_iter = label_map.begin(); lt_iter != label_map.end(); lt_iter++) {
     VarLabel* var = VarLabel::find(lt_iter->first);
     
     if (var == nullptr) {
       throw ProblemSetupException(lt_iter->first + " variable not found to checkpoint.",__FILE__, __LINE__);
     }
     
     saveItem.label = var;
     saveItem.matlSet.clear();
     
     map<int, ConsecutiveRangeSet>::iterator map_iter;
     for (map_iter = lt_iter->second.begin(); map_iter != lt_iter->second.end(); ++map_iter) {
       
       saveItem.setMaterials(map_iter->first, map_iter->second, m_prevMatls, m_prevMatlSet);

       if (string(var->getName()) == delT_name) {
         hasDelT = true;
       }
     }
     
     // Skip this variable if the default behavior of variable has
     // been overwritten.  For example ignore checkpointing
     // PerPatch<FileInfo> variable
     bool skipVar = ( notCheckPointVars.count(saveItem.label->getName() ) > 0 );
     
     if( !skipVar ) {
       if ( saveItem.label->typeDescription()->isReductionVariable() ) {
         m_checkpointReductionLabels.push_back(saveItem);
       } else {
         m_checkpointLabels.push_back(saveItem);
       }
     }
   }


   if (!hasDelT) {
     VarLabel* var = VarLabel::find(delT_name);
     if (var == nullptr) {
       throw ProblemSetupException("delT variable not found to checkpoint.",__FILE__, __LINE__);
     }
     
     saveItem.label = var;
     saveItem.matlSet.clear();
     ConsecutiveRangeSet globalMatl("-1");
     saveItem.setMaterials(-1,globalMatl, m_prevMatls, m_prevMatlSet);
     ASSERT(saveItem.label->typeDescription()->isReductionVariable());
     m_checkpointReductionLabels.push_back(saveItem);
   }     
}

//______________________________________________________________________
//
void
DataArchiver::SaveItem::setMaterials(int level, 
                                     const ConsecutiveRangeSet& matls,
                                     ConsecutiveRangeSet& prevMatls,
                                     MaterialSetP& prevMatlSet)
{
  // reuse material sets when the same set of materials is used for
  // different SaveItems in a row -- easier than finding all reusable
  // material set, but effective in many common cases.
  if( (prevMatlSet != nullptr) && (matls == prevMatls) ) {
    matlSet[level] = prevMatlSet;
  }
  else {
    MaterialSetP& m = matlSet[level];
    m = scinew MaterialSet();
    vector<int> matlVec;
    matlVec.reserve(matls.size());

    for (ConsecutiveRangeSet::iterator crs_iter = matls.begin();
	 crs_iter != matls.end(); ++crs_iter) {
      matlVec.push_back(*crs_iter);
    }

    m->addAll(matlVec);
    prevMatlSet = m;
    prevMatls = matls;
  }
}

//______________________________________________________________________
//  Find the materials to output on this level for this saveItem
const MaterialSubset*
DataArchiver::SaveItem::getMaterialSubset(const Level* level)
{
  // search done by absolute level, or relative to end of levels (-1
  // finest, -2 second finest,...)
  map<int, MaterialSetP>::iterator iter = matlSet.end();
  const MaterialSubset* var_matls = nullptr;
  
  if (level) {
    int L_index = level->getIndex();
    int maxLevels = level->getGrid()->numLevels();
    
    iter = matlSet.find( L_index );
    
    if (iter == matlSet.end()){
      iter = matlSet.find( L_index - maxLevels );
    }
    
    if (iter == matlSet.end()) {
      iter = matlSet.find(ALL_LEVELS);
    }
    
    if (iter != matlSet.end()) {
      var_matls = iter->second.get_rep()->getUnion();
    }
  }
  else { // reductions variables that are level independent
    map<int, MaterialSetP>::iterator iter;
    for (iter = matlSet.begin(); iter != matlSet.end(); ++iter) {
      var_matls = getMaterialSet(iter->first)->getUnion();
      break;
    }
  }
  return var_matls;
}


//______________________________________________________________________
//
bool
DataArchiver::needRecompile(const GridP& /*grid*/)
{
  bool retVal = false;
  
#ifdef HAVE_PIDX
  retVal = (retVal || m_pidx_need_to_recompile || m_pidx_restore_nth_rank);
#endif         

  return retVal;
}

//______________________________________________________________________
//
void
DataArchiver::recompile( const GridP& grid )
{

#ifdef HAVE_PIDX
  if( m_pidx_requested_nth_rank > 1 ) {      
    if( m_pidx_restore_nth_rank ) {
      proc0cout << "This is the time step following a checkpoint - "
		<< "need to put the task graph back with a recompile - "
		<< "setting nth output to 1\n";
      m_loadBalancer->setNthRank( 1 );
      m_loadBalancer->possiblyDynamicallyReallocate( grid,
						     LoadBalancer::regrid );
      setSaveAsPIDX();
      m_pidx_restore_nth_rank = false;
    }
        
    if( m_pidx_need_to_recompile ) {
      // Don't need to recompile on the next time step as it will
      // happen on this one.  However, the nth rank value will
      // need to be restored after this time step, so set
      // pidx_restore_nth_rank to true.
      m_pidx_need_to_recompile = false;
      m_pidx_restore_nth_rank = true;
    }
  }
#endif
}

//______________________________________________________________________
//
string
DataArchiver::TranslateVariableType( string type, bool isThisCheckpoint )
{
  if ( m_outputDoubleAsFloat && !isThisCheckpoint ) {
    if      (type=="CCVariable<double>"  ) return "CCVariable<float>";
    else if (type=="NCVariable<double>"  ) return "NCVariable<float>";
    else if (type=="SFCXVariable<double>") return "SFCXVariable<float>";
    else if (type=="SFCYVariable<double>") return "SFCYVariable<float>";
    else if (type=="SFCZVariable<double>") return "SFCZVariable<float>";
    else if (type=="ParticleVariable<double>") return "ParticleVariable<float>";
  }
  return type;
}

bool
DataArchiver::isLabelSaved( const string & label ) const
{
  if(m_outputInterval == 0.0 && m_outputTimeStepInterval == 0) {
    return false;
  }

  list<SaveNameItem>::const_iterator iter;
  for( iter = m_saveLabelNames.begin(); iter != m_saveLabelNames.end(); ++iter ) {
    if( iter->labelName == label ) {
      return true;
    }
  }
  return false;
}

//__________________________________
// Allow the component to set the output interval
void
DataArchiver::setOutputInterval( double newinv )
{
  if (m_outputInterval != newinv)
  {
    m_outputInterval = newinv;
    m_outputTimeStepInterval = 0;
    m_nextOutputTime = 0.0;
  }
}

//__________________________________
// Allow the component to set the output timestep interval
void
DataArchiver::setOutputTimeStepInterval( int newinv )
{
  if (m_outputTimeStepInterval != newinv)
  {
    m_outputTimeStepInterval = newinv;
    m_outputInterval = 0;
    m_nextOutputTime = 0.0;
  }
}

//__________________________________
// Allow the component to set the checkpoint interval
void
DataArchiver::setCheckpointInterval( double newinv )
{
  if (m_checkpointInterval != newinv)
  {
    m_checkpointInterval = newinv;
    m_checkpointTimeStepInterval = 0;
    m_nextCheckpointTime = 0.0;

    // If needed create checkpoints/index.xml
    if( !m_checkpointsDir.exists() )
    {
      if( d_myworld->myRank() == 0) {
        m_checkpointsDir = m_dir.createSubdir("checkpoints");
        createIndexXML(m_checkpointsDir);
      }
    }

    // Sync up before every rank can use the checkpoints dir
    Uintah::MPI::Barrier(d_myworld->getComm());
  }
}

//__________________________________
// Allow the component to set the checkpoint timestep interval
void
DataArchiver::setCheckpointTimeStepInterval( int newinv )
{
  if (m_checkpointTimeStepInterval != newinv)
  {
    m_checkpointTimeStepInterval = newinv;
    m_checkpointInterval = 0;
    m_nextCheckpointTime = 0.0;

    // If needed create checkpoints/index.xml
    if( !m_checkpointsDir.exists())
    {
      if( d_myworld->myRank() == 0) {
        m_checkpointsDir = m_dir.createSubdir("checkpoints");
        createIndexXML(m_checkpointsDir);
      }
    }

    // Sync up before every rank can use the checkpoints dir
    Uintah::MPI::Barrier(d_myworld->getComm());
  }
}

//______________________________________________________________________
//  This will copy the portions of the timestep.xml from the old uda
//  to the new uda.  Specifically, the sections related to the
//  components.
void
DataArchiver::copy_outputProblemSpec( Dir & fromDir, Dir & toDir )
{
  int dir_timestep = getTimeStepTopLevel();
  
  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << dir_timestep;
  
  // define the from/to directories & files
  string fromPath = fromDir.getName()+"/"+tname.str();
  Dir myFromDir   = Dir(fromPath);
  string fromFile = fromPath + "/timestep.xml";
  
  string toPath   = toDir.getName() + "/" + tname.str();
  Dir myToDir     = Dir( toPath );
  string toFile   = toPath + "/timestep.xml";  
  
  //__________________________________
  //  loop over the blocks in timestep.xml
  //  and copy the component related nodes 
  ProblemSpecP inputDoc = loadDocument( fromFile );

  for (ProblemSpecP ps = inputDoc->getFirstChild(); ps != nullptr; ps = ps->getNextSibling()) {
    string nodeName = ps->getNodeName();

    if (nodeName == "Meta" || nodeName == "Time" || nodeName == "Grid" || nodeName == "Data") {
      continue;
    }
    cout << "   Now copying the XML node (" << setw(20) << nodeName << ")" << " from: " << fromFile << " to: " << toFile << "\n";
    copySection( myFromDir,  myToDir, "timestep.xml", nodeName );
  }
} 

//______________________________________________________________________
// If your using PostProcessUda then use its mapping 
int
DataArchiver::getTimeStepTopLevel()
{
  const int timeStep = m_application->getTimeStep();

  if ( m_doPostProcessUda ) {
    return m_restartTimeStepIndicies[ timeStep ];
  }
  else {
    return timeStep;
  }
}

//______________________________________________________________________
// Called by In-situ VisIt to dump a time step's data.
void
DataArchiver::outputTimeStep( const GridP& grid,
                              SchedulerP& sched )
{
  int proc = d_myworld->myRank();

  DataWarehouse* oldDW = sched->get_dw(0);
  DataWarehouse* newDW = sched->getLastDW();
  
  const int timeStep = m_application->getTimeStep();

  // Save the var so to return to the normal output schedule.
  int nextOutputTimeStep = m_nextOutputTimeStep;
  int outputTimeStepInterval = m_outputTimeStepInterval;

  m_nextOutputTimeStep = timeStep;
  m_outputTimeStepInterval = 1;

  // Set up the inital bits including the flag m_isOutputTimeStep
  // which triggers most actions.
  beginOutputTimeStep( grid);

  // Updaate the main xml file and write the xml file for this
  // timestep.
  writeto_xml_files( grid );

  // For each level get the patches associated with this processor and
  // save the requested output variables.
  for( int i = 0; i < grid->numLevels(); ++i ) {
    const LevelP& level = grid->getLevel( i );

    const PatchSet* patches = m_loadBalancer->getOutputPerProcessorPatchSet(level);

    outputVariables( nullptr, patches->getSubset(proc), nullptr, oldDW, newDW, OUTPUT );
    outputVariables( nullptr, patches->getSubset(proc), nullptr, oldDW, newDW, CHECKPOINT );
  }

  // Restore the timestep vars so to return to the normal output
  // schedule.
  m_nextOutputTimeStep = nextOutputTimeStep;
  m_outputTimeStepInterval = outputTimeStepInterval;

  m_isOutputTimeStep = false;
  m_isCheckpointTimeStep = false;
}

//______________________________________________________________________
// Called by In-situ VisIt to dump a checkpoint.
//
void
DataArchiver::checkpointTimeStep( const GridP& grid,
                                  SchedulerP& sched )
{
  int proc = d_myworld->myRank();

  DataWarehouse* oldDW = sched->get_dw(0);
  DataWarehouse* newDW = sched->getLastDW();

  const int timeStep = m_application->getTimeStep();

  // Save the vars so to return to the normal output schedule.
  int nextCheckpointTimeStep = m_nextCheckpointTimeStep;
  int checkpointTimeStepInterval = m_checkpointTimeStepInterval;

  m_nextCheckpointTimeStep = timeStep;
  m_checkpointTimeStepInterval = 1;

  // If needed create checkpoints/index.xml
  if( !m_checkpointsDir.exists()) {
    if( proc == 0) {
      m_checkpointsDir = m_dir.createSubdir("checkpoints");
      createIndexXML( m_checkpointsDir );
    }
  }

  // Sync up before every rank can use the checkpoints dir
  Uintah::MPI::Barrier( d_myworld->getComm() );

  // Set up the inital bits including the flag m_isCheckpointTimeStep
  // which triggers most actions.
  beginOutputTimeStep( grid );

  // Updaate the main xml file and write the xml file for this
  // timestep.
  writeto_xml_files( grid );

  // For each level get the patches associated with this processor and
  // save the requested output variables.
  for( int i = 0; i < grid->numLevels(); ++i ) {
    const LevelP& level = grid->getLevel( i );

    const PatchSet* patches = m_loadBalancer->getOutputPerProcessorPatchSet(level);

    outputVariables( nullptr, patches->getSubset(proc),
		     nullptr, oldDW, newDW, CHECKPOINT );
  }

  // Restore the vars so to return to the normal output schedule.
  m_nextCheckpointTimeStep = nextCheckpointTimeStep;
  m_checkpointTimeStepInterval = checkpointTimeStepInterval;

  m_isOutputTimeStep = false;
  m_isCheckpointTimeStep = false;
}

//______________________________________________________________________
// Called by In-situ VisIt to dump a checkpoint.
//
void
DataArchiver::setElapsedWallTime( double val )
{
  // When using the wall time, rank 0 determines the wall time and
  // sends it to all other ranks.
  Uintah::MPI::Bcast( &val, 1, MPI_DOUBLE, 0, d_myworld->getComm() );
  
  m_elapsedWallTime = val;
};

//______________________________________________________________________
//
//
void
DataArchiver::saveSVNinfo()
{
  string svn_diff_file =
    string( sci_getenv("SCIRUN_OBJDIR") ) + "/svn_diff.txt";
  
  if( !validFile( svn_diff_file ) ) {
    cout << "\n"
	 << "WARNING: 'svn diff' file '" << svn_diff_file
	 << "' does not appear to exist!\n"
	 << "\n";
  } 
  else {
    string svn_diff_out = m_dir.getName() + "/svn_diff.txt";
    string svn_diff_on = string( sci_getenv("SCIRUN_OBJDIR") ) + "/.do_svn_diff";
    if( !validFile( svn_diff_on ) ) {
      cout << "\n"
	   << "WARNING: Adding 'svn diff' file to UDA, "
	   << "but AUTO DIFF TEXT CREATION is OFF!\n"
	   << "         svn_diff.txt may be out of date!  "
	   << "Saving as 'possible_svn_diff.txt'.\n"
	   << "\n";
      
      svn_diff_out = m_dir.getName() + "/possible_svn_diff.txt";
    }

    copyFile( svn_diff_file, svn_diff_out );
  }
}

//______________________________________________________________________
//
// Verifies that all processes can see the same file system (as rank 0).
//
void
DataArchiver::setupSharedFileSystem()
{
  Timers::Simple timer;
  timer.start();

  // Verify that all MPI processes can see the common file system (with rank 0).
  string fs_test_file_name;
  if( d_myworld->myRank() == 0 ) {

    m_writeMeta = true;
    // Create a unique file name, using hostname + pid
    char hostname[ MAXHOSTNAMELEN ];
    if( gethostname( hostname, MAXHOSTNAMELEN ) != 0 ) {
      strcpy( hostname, "hostname-unknown" );
    }

    ostringstream test_filename_stream;
    test_filename_stream << "sus_filesystem_test-" << hostname
			 << "-" << getpid();

    // Create the test file...
    FILE * tmpout = fopen( test_filename_stream.str().c_str(), "w" );
    if( !tmpout ) {
      throw ErrnoException("fopen failed for " + test_filename_stream.str(),
			   errno, __FILE__, __LINE__ );
    }
    fprintf( tmpout, "\n" ); // Test writing to file...
    if( fflush( tmpout ) != 0 ) {
      throw ErrnoException( "fflush", errno, __FILE__, __LINE__ );
    }
    if( fsync( fileno( tmpout ) ) != 0 ) { // Test syncing a file.
      throw ErrnoException( "fsync", errno, __FILE__, __LINE__ );
    }
    if( fclose(tmpout) != 0) { // Test closing the file.
      throw ErrnoException( "fclose", errno, __FILE__, __LINE__ );
    }

    // While the following has never before been necessary, it turns out that
    // the "str()" operator on an ostringstream creates a temporary buffer
    // that can be deleted at any time and so using ".c_str()" on it may return
    // garbage.  To avoid this, we need to copy the ".str()" output into our
    // own string, and then use the ".c_str()" on that non-temporary string.
    const string temp_string = test_filename_stream.str();

    const char* outbuf = temp_string.c_str();
    int         outlen = (int)strlen( outbuf );

    // Broadcast test filename length, and then broadcast the actual name.
    Uintah::MPI::Bcast( &outlen, 1, MPI_INT, 0, d_myworld->getComm() );
    Uintah::MPI::Bcast( const_cast<char*>(outbuf), outlen, MPI_CHAR, 0,
			d_myworld->getComm() );
    fs_test_file_name = test_filename_stream.str();
  } 
  else {
    m_writeMeta = false; // Only rank 0 will emit meta data...

    // All other ranks receive from rank 0 (code above) the length,
    // and then name of the file that we are going to look for...
    int inlen;
    Uintah::MPI::Bcast( &inlen, 1, MPI_INT, 0, d_myworld->getComm() );
    char * inbuf = scinew char[ inlen + 1 ];
    Uintah::MPI::Bcast( inbuf, inlen, MPI_CHAR, 0, d_myworld->getComm() );
    inbuf[ inlen ]='\0';
    fs_test_file_name = inbuf;

    delete[] inbuf;       
  }

  if( d_myworld->myRank() != 0 ) { // Make sure everyone else can see the temp file...

    struct stat st;
    int s = stat( fs_test_file_name.c_str(), &st );

    if( ( s != 0 ) || !S_ISREG( st.st_mode ) ) {
      cerr << "Stat'ing of file: " << fs_test_file_name
	   << " failed with errno = " << errno << "\n";
      throw ErrnoException( "stat", errno, __FILE__, __LINE__ );
    }
  }

  Uintah::MPI::Barrier(d_myworld->getComm()); // Wait until everyone has check for the file before proceeding.

  if( m_writeMeta ) {

    int s = unlink( fs_test_file_name.c_str() ); // Remove the tmp file...
    if(s != 0) {
      cerr << "Cannot unlink file: " << fs_test_file_name << '\n';
      throw ErrnoException("unlink", errno, __FILE__, __LINE__);
    }

    makeVersionedDir();
    // Send UDA name to all other ranks.
    string udadirname = m_dir.getName();
    
    // Broadcast uda dir name length, and then broadcast the actual name.
    const char* outbuf = udadirname.c_str();
    int         outlen = (int)strlen(outbuf);

    Uintah::MPI::Bcast( &outlen, 1, MPI_INT, 0, d_myworld->getComm() );
    Uintah::MPI::Bcast( const_cast<char*>(outbuf), outlen, MPI_CHAR, 0,
			d_myworld->getComm() );
  }
  else {

    // Receive the name of the UDA from rank 0...
    int inlen;
    Uintah::MPI::Bcast( &inlen, 1, MPI_INT, 0, d_myworld->getComm() );
    char * inbuf = scinew char[ inlen+1 ];
    Uintah::MPI::Bcast( inbuf, inlen, MPI_CHAR, 0, d_myworld->getComm() );
    inbuf[ inlen ]='\0';

    m_dir = Dir( inbuf );
    delete[] inbuf;
  }

  if( d_myworld->myRank() == 0 ) {
    cerr << "Verified shared file system in " << timer().seconds()
	 << " seconds.\n";
  }

} // end setupSharedFileSystem()

//______________________________________________________________________
//
// setupLocalFileSystems()
//
// This is the old method of checking for shared vs local file systems
// and determining which node(s) will write the UDA meta data.  Rank 0
// creates a file name, sends it to all other ranks.  All other ranks
// use that basename to create subsequent files ("basename-rank").
// Then all ranks (except 0) start looking for these files in order
// (starting one past their rank).  If a file is found, it means that
// the rank is on a node that has another (lower) rank - and that
// lower rank will do the writing.
//
void
DataArchiver::setupLocalFileSystems()
{
  Timers::Simple timer;
  timer.start();

  // See how many shared filesystems that we have
  string basename;
  if( d_myworld->myRank() == 0 ) {
    // Create a unique string, using hostname+pid
    char* base = strdup(m_filebase.c_str());
    char* p = base+strlen(base);
    for(;p>=base;p--) {
      if(*p == '/') {
        *++p=0; // keep trailing slash
        break;
      }
    }

    if(*base) {
      free(base);
      base = strdup(".");
    }

    char hostname[MAXHOSTNAMELEN];
    if(gethostname(hostname, MAXHOSTNAMELEN) != 0)
      strcpy(hostname, "unknown???");
    ostringstream ts;
    ts << base << "-" << hostname << "-" << getpid();
    if (*base) {
      free(base);
    }

    string test_string = ts.str();
    const char* outbuf = test_string.c_str();
    int outlen = (int)strlen(outbuf);

    Uintah::MPI::Bcast(&outlen, 1, MPI_INT, 0, d_myworld->getComm());
    Uintah::MPI::Bcast(const_cast<char*>(outbuf), outlen, MPI_CHAR, 0,
              d_myworld->getComm());
    basename = test_string;
  }
  else {
    int inlen;
    Uintah::MPI::Bcast(&inlen, 1, MPI_INT, 0, d_myworld->getComm());
    char* inbuf = scinew char[inlen+1];
    Uintah::MPI::Bcast(inbuf, inlen, MPI_CHAR, 0, d_myworld->getComm());
    inbuf[inlen]='\0';
    basename=inbuf;
    delete[] inbuf;       
  }
  // Create a file, of the name p0_hostname-p0_pid-processor_number
  ostringstream myname;
  myname << basename << "-" << d_myworld->myRank() << ".tmp";
  string fname = myname.str();

  // This will be an empty file, everything is encoded in the name anyway
  FILE* tmpout = fopen(fname.c_str(), "w");
  if(!tmpout) {
    throw ErrnoException("fopen failed for " + fname, errno,
			 __FILE__, __LINE__);
  }
  fprintf(tmpout, "\n");
  if(fflush(tmpout) != 0) {
    throw ErrnoException("fflush", errno, __FILE__, __LINE__);
  }
  if(fsync(fileno(tmpout)) != 0) {
    throw ErrnoException("fsync", errno, __FILE__, __LINE__);
  }
  if(fclose(tmpout) != 0) {
    throw ErrnoException("fclose", errno, __FILE__, __LINE__);
  }
  Uintah::MPI::Barrier(d_myworld->getComm());
  // See who else we can see
  m_writeMeta=true;
  int i;
  for(i=0;i<d_myworld->myRank();i++) {
    ostringstream name;
    name << basename << "-" << i << ".tmp";
    struct stat st;
    int s=stat(name.str().c_str(), &st);
    if(s == 0 && S_ISREG(st.st_mode)) {
      // File exists, we do NOT need to emit metadata
      m_writeMeta=false;
      break;
    }
    else if(errno != ENOENT) {
      cerr << "Cannot stat file: " << name.str() << ", errno=" << errno << '\n';
      throw ErrnoException("stat", errno, __FILE__, __LINE__);
    }
  }

  Uintah::MPI::Barrier( d_myworld->getComm() );

  if( m_writeMeta ) {
    makeVersionedDir();
    string fname = myname.str();
    FILE* tmpout = fopen(fname.c_str(), "w");
    if(!tmpout) {
      throw ErrnoException("fopen", errno, __FILE__, __LINE__);
    }
    string dirname = m_dir.getName();
    fprintf(tmpout, "%s\n", dirname.c_str());
    if(fflush(tmpout) != 0) {
      throw ErrnoException("fflush", errno, __FILE__, __LINE__);
    }
#if defined(__APPLE__)
    if(fsync(fileno(tmpout)) != 0) {
      throw ErrnoException("fsync", errno, __FILE__, __LINE__);
    }
#elif !defined(__bgq__) // __bgq__ is defined on Blue Gene Q computers...
    if(fdatasync(fileno(tmpout)) != 0) {
      throw ErrnoException("fdatasync", errno, __FILE__, __LINE__);
    }
#endif
    if(fclose(tmpout) != 0) {
      throw ErrnoException("fclose", errno, __FILE__, __LINE__);
    }
  }
  
  Uintah::MPI::Barrier(d_myworld->getComm());

if(!m_writeMeta) {
    ostringstream name;
    name << basename << "-" << i << ".tmp";
    ifstream in(name.str().c_str()); 
    if (!in) {
      throw InternalError("DataArchiver::initializeOutput(): The file \"" +
                          name.str() +
			  "\" not found on second pass for filesystem discovery!",
                          __FILE__, __LINE__);
    }
    string dirname;
    in >> dirname;
    m_dir = Dir( dirname );
  }

  int count = m_writeMeta ? 1 : 0;
  int nunique;
  // This is an AllReduce, not a reduce.  This is necessary to
  // ensure that all processors wait before they remove the tmp files
  Uintah::MPI::Allreduce(&count, &nunique, 1, MPI_INT, MPI_SUM,
                d_myworld->getComm());
  if( d_myworld->myRank() == 0 ) {
    cerr << "Discovered " << nunique << " unique filesystems in "
	 << timer().seconds() << " seconds\n";
  }
  // Remove the tmp files...
  int s = unlink(myname.str().c_str());
  if( s != 0 ) {
    cerr << "Cannot unlink file: " << myname.str() << '\n';
    throw ErrnoException( "unlink", errno, __FILE__, __LINE__ );
  }
}
