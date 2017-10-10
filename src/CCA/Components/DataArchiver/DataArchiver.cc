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
#include <CCA/Ports/LoadBalancerPort.h>
#include <CCA/Ports/OutputContext.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

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

bool DataArchiver::d_wereSavesAndCheckpointsInitialized = false;

DataArchiver::DataArchiver(const ProcessorGroup* myworld, int udaSuffix)
  : UintahParallelComponent(myworld),
    d_udaSuffix(udaSuffix)
{
  d_isOutputTimestep      = false;
  d_isCheckpointTimestep  = false;
  d_saveParticleVariables = false;
  d_saveP_x               = false;
  d_particlePositionName  = "p.x";
  d_usingReduceUda        = false;
  d_outputFileFormat      = UDA;
  //d_currentTime=-1;
  //d_currentTimestep=-1;

  d_XMLIndexDoc = nullptr;
  d_CheckpointXMLIndexDoc = nullptr;

  d_outputDoubleAsFloat = false;

  d_fileSystemRetrys = 10;
  d_numLevelsInOutput = 0;

  d_writeMeta = false;
}

DataArchiver::~DataArchiver()
{
}

//______________________________________________________________________
//
void
DataArchiver::problemSetup( const ProblemSpecP    & params,
			    const ProblemSpecP    & restart_prob_spec,
                                  SimulationState * state )
{
  dbg << "Doing ProblemSetup \t\t\t\tDataArchiver\n";

  d_sharedState = state;
  d_upsFile = params;
  ProblemSpecP p = params->findBlock("DataArchiver");
  
  if( restart_prob_spec ) {

    ProblemSpecP insitu_ps = restart_prob_spec->findBlock( "InSitu" );

    if( insitu_ps != nullptr ) {

      bool haveModifiedVars;
      
      insitu_ps = insitu_ps->get("haveModifiedVars", haveModifiedVars);

      d_sharedState->haveModifiedVars( haveModifiedVars );

      if( haveModifiedVars )
      {
	std::stringstream tmpstr;
	tmpstr << "DataArchiver found previously modified variables that "
	       << "have not been merged into the checkpoint restart "
	       << "input.xml file from the from index.xml file. " << std::endl
	       << "The modified variables can be found in the "
	       << "index.xml file under the 'InSitu' block." << std::endl
	       << "Once merged, change the variable 'haveModifiedVars' in "
	       << "the 'InSitu' block in the checkpoint restart timestep.xml "
	       << "file to 'false'";
	
	throw ProblemSetupException( tmpstr.str() ,__FILE__, __LINE__);
      }
      else
      {
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
  if( type == "pidx" || type == "PIDX" ) {
    d_outputFileFormat = PIDX;
    d_PIDX_flags.problemSetup( p );
    //d_PIDX_flags.print();
  }

  d_outputDoubleAsFloat = p->findBlock("outputDoubleAsFloat") != nullptr;

  // set to false if restartSetup is called - we can't do it there
  // as the first timestep doesn't have any tasks
  d_outputInitTimestep = p->findBlock("outputInitTimestep") != nullptr;
  
  // problemSetup is called again from the Switcher to reset vars (and
  // frequency) it wants to save DO NOT get it again.  Currently the
  // directory won't change mid-run, so calling problemSetup will not
  // change the directory.  What happens then, is even if a switched
  // component wants a different uda name, it will not get one until
  // sus restarts (i.e., when you switch, component 2's data dumps
  // will be in whichever uda started sus.), which is not optimal.  So
  // we disable this feature until we can make the DataArchiver make a
  // new directory mid-run.
  if (d_filebase == "") {
    p->require("filebase", d_filebase);
  }

  // Get output timestep interval, or time interval info:
  d_outputInterval = 0;
  if( !p->get( "outputTimestepInterval", d_outputTimestepInterval ) ) {
    d_outputTimestepInterval = 0;
  }
  
  if ( !p->get("outputInterval", d_outputInterval) && d_outputTimestepInterval == 0 ) {
    d_outputInterval = 0.0; // default
  }

  if ( d_outputInterval > 0.0 && d_outputTimestepInterval > 0 ) {
    throw ProblemSetupException("Use <outputInterval> or <outputTimestepInterval>, not both",__FILE__, __LINE__);
  }

  if ( !p->get("outputLastTimestep", d_outputLastTimestep) ) {
    d_outputLastTimestep = false; // default
  }

  // set default compression mode - can be "tryall", "gzip", "rle",
  // "rle, gzip", "gzip, rle", or "none"
  string defaultCompressionMode = "";
  if (p->get("compression", defaultCompressionMode)) {
    VarLabel::setDefaultCompressionMode(defaultCompressionMode);
  }

  if (params->findBlock("ParticlePosition")) {
    params->findBlock("ParticlePosition")->getAttribute("label",d_particlePositionName);
  }

  //__________________________________
  // parse the variables to be saved
  d_saveLabelNames.clear(); // we can problemSetup multiple times on a component Switch, clear the old ones.
  map<string, string> attributes;
  SaveNameItem saveItem;
  ProblemSpecP save = p->findBlock("save");

  if( save == nullptr ) {
    // If no <save> labels were specified, make sure that an output
    // time interval is not specified...
    if( d_outputInterval > 0.0 || d_outputTimestepInterval > 0 ) {
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
    if (saveItem.labelName == d_particlePositionName || saveItem.labelName == "p.xx") {
      d_saveP_x = true;
    }

    string::size_type pos = saveItem.labelName.find("p.");
    if ( pos != string::npos &&  saveItem.labelName != d_particlePositionName) {
      d_saveParticleVariables = true;
    }

    d_saveLabelNames.push_back( saveItem );

    save = save->findNextBlock( "save" );
  }
  
  if(d_saveP_x == false && d_saveParticleVariables == true) {
    throw ProblemSetupException(" You must save " + d_particlePositionName + " when saving other particle variables", __FILE__, __LINE__);
  }     

  //__________________________________
  // get checkpoint information
  d_checkpointInterval         = 0.0;
  d_checkpointTimestepInterval = 0;
  d_checkpointWalltimeStart    = 0;
  d_checkpointWalltimeInterval = 0;
  d_checkpointCycle            = 2; /* 2 is the smallest number that is safe
                                    (always keeping an older copy for backup) */
  d_checkpointLastTimestep     = false;
  
  ProblemSpecP checkpoint = p->findBlock( "checkpoint" );
  if( checkpoint != nullptr ) {

    string interval, timestepInterval, walltimeStart, walltimeInterval,
      walltimeStartHours, walltimeIntervalHours, cycle, lastTimestep;

    attributes.clear();
    checkpoint->getAttributes( attributes );

    interval              = attributes[ "interval" ];
    timestepInterval      = attributes[ "timestepInterval" ];
    walltimeStart         = attributes[ "walltimeStart" ];
    walltimeInterval      = attributes[ "walltimeInterval" ];
    walltimeStartHours    = attributes[ "walltimeStartHours" ];
    walltimeIntervalHours = attributes[ "walltimeIntervalHours" ];
    cycle                 = attributes[ "cycle" ];
    lastTimestep          = attributes[ "lastTimestep" ];

    if( interval != "" ) {
      d_checkpointInterval = atof( interval.c_str() );
    }
    if( timestepInterval != "" ) {
      d_checkpointTimestepInterval = atoi( timestepInterval.c_str() );
    }
    if( walltimeStart != "" ) {
      d_checkpointWalltimeStart = atoi( walltimeStart.c_str() );
    }      
    if( walltimeInterval != "" ) {
      d_checkpointWalltimeInterval = atoi( walltimeInterval.c_str() );
    }
    if( walltimeStartHours != "" ) {
      d_checkpointWalltimeStart = atof( walltimeStartHours.c_str() ) * 3600.0;
    }      
    if( walltimeIntervalHours != "" ) {
      d_checkpointWalltimeInterval = atof( walltimeIntervalHours.c_str() ) * 3600.0;
    }
    if( cycle != "" ) {
      d_checkpointCycle = atoi( cycle.c_str() );
    }

    if( lastTimestep == "true" ) {
      d_checkpointLastTimestep = true;
    }

    // Verify that an interval was specified:
    if( interval == "" && timestepInterval == "" &&
	walltimeInterval == "" && walltimeIntervalHours == "" ) {
      throw ProblemSetupException( "ERROR: \n  <checkpoint> must specify either interval, timestepInterval, walltimeInterval",
                                   __FILE__, __LINE__ );
    }
  }

  // Can't use both checkpointInterval and checkpointTimestepInterval.
  if (d_checkpointInterval > 0.0 && d_checkpointTimestepInterval > 0) {
    throw ProblemSetupException("Use <checkpoint interval=...> or <checkpoint timestepInterval=...>, not both",
                                __FILE__, __LINE__);
  }
  // Can't have a walltimeStart without a walltimeInterval.
  if (d_checkpointWalltimeStart > 0.0 && d_checkpointWalltimeInterval == 0) {
    throw ProblemSetupException("<checkpoint walltimeStart must have a corresponding walltimeInterval",
                                __FILE__, __LINE__);
  }

  d_lastTimestepLocation   = "invalid";
  d_isOutputTimestep       = false;

  // Set up the next output and checkpoint time. Always output the
  // first timestep or the inital timestep.
  d_nextOutputTime     = 0.0; 
  d_nextOutputTimestep = d_outputInitTimestep ? 0 : 1;

  d_nextCheckpointTime     = d_checkpointInterval;
  d_nextCheckpointTimestep = d_checkpointTimestepInterval + 1;
  d_nextCheckpointWalltime = d_checkpointWalltimeStart;

  //__________________________________
  // 
  if ( d_checkpointInterval > 0 ) {
    proc0cout << "Checkpointing:" << std::setw(16) << " Every "
	      << d_checkpointInterval << " physical seconds.\n";
  }
  if ( d_checkpointTimestepInterval > 0 ) {
    proc0cout << "Checkpointing:" << std::setw(16)<< " Every "
	      << d_checkpointTimestepInterval << " timesteps.\n";
  }
  if ( d_checkpointWalltimeInterval > 0 ) {
    proc0cout << "Checkpointing:" << std::setw(16)<< " Every "
	      << d_checkpointWalltimeInterval << " wall clock seconds,"
              << " starting after " << d_checkpointWalltimeStart << " seconds.\n";
  }
  
#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( d_sharedState->getVisIt() && !initialized ) {
    d_sharedState->d_debugStreams.push_back( &dbg  );
#ifdef HAVE_PIDX
    d_sharedState->d_debugStreams.push_back( &dbgPIDX );
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
  dbg << "Doing outputProblemSpec \t\t\t\tDataArchiver\n";

  ProblemSpecP root = root_ps->getRootNode();

  if( d_sharedState->haveModifiedVars() ) {

    ProblemSpecP is_ps = root->findBlockWithOutAttribute( "InSitu" );

    if( is_ps == nullptr )
      is_ps = root->appendChild("InSitu");

    is_ps->appendElement("haveModifiedVars", d_sharedState->haveModifiedVars());
  }
}


//______________________________________________________________________
//
void
DataArchiver::initializeOutput( const ProblemSpecP & params )
{
  if( d_outputInterval             == 0.0 && 
      d_outputTimestepInterval     == 0   && 
      d_checkpointInterval         == 0.0 && 
      d_checkpointTimestepInterval == 0   && 
      d_checkpointWalltimeInterval == 0 ) {
    return;
  }

  if( d_sharedState->getUseLocalFileSystems() ) {
    setupLocalFileSystems();
  }
  else {
    setupSharedFileSystem();
  }
  // Wait for all ranks to finish verifying shared file system....
  Uintah::MPI::Barrier(d_myworld->getComm());

  if (d_writeMeta) {

    saveSVNinfo();
    // Create index.xml:
    string inputname = d_dir.getName()+"/input.xml";
    params->output( inputname.c_str() );

    /////////////////////////////////////////////////////////
    // Save the original .ups file in the UDA...
    //     FIXME: might want to avoid using 'system' copy which the
    //     below uses...  If so, we will need to write our own
    //     (simple) file reader and writer routine.

    cout << "Saving original .ups file in UDA...\n";
    Dir ups_location( pathname( params->getFile() ) );
    ups_location.copy( basename( params->getFile() ), d_dir );

    //
    /////////////////////////////////////////////////////////

    createIndexXML(d_dir);
   
    // create checkpoints/index.xml (if we are saving checkpoints)
    if ( d_checkpointInterval         > 0.0 || 
         d_checkpointTimestepInterval > 0   || 
         d_checkpointWalltimeInterval > 0 ) {
      d_checkpointsDir = d_dir.createSubdir("checkpoints");
      createIndexXML(d_checkpointsDir);
    }
  }
  else {
    d_checkpointsDir = d_dir.getSubdir("checkpoints");
  }

  // Sync up before every rank can use the base dir.
  Uintah::MPI::Barrier(d_myworld->getComm());
} // end initializeOutput()

//______________________________________________________________________
// to be called after problemSetup and initializeOutput get called
void
DataArchiver::restartSetup( Dir    & restartFromDir,
                            int      startTimestep,
                            int      timestep,
                            double   time,
                            bool     fromScratch,
                            bool     removeOldDir )
{
  d_outputInitTimestep = false;

  if( d_writeMeta && !fromScratch ) {
    // partial copy of dat files
    copyDatFiles( restartFromDir, d_dir, startTimestep, timestep, removeOldDir );

    copySection( restartFromDir, d_dir, "index.xml", "restarts" );
    copySection( restartFromDir, d_dir, "index.xml", "variables" );
    copySection( restartFromDir, d_dir, "index.xml", "globals" );

    // partial copy of index.xml and timestep directories and
    // similarly for checkpoints
    copyTimesteps(restartFromDir, d_dir, startTimestep, timestep, removeOldDir);

    Dir checkpointsFromDir = restartFromDir.getSubdir("checkpoints");
    bool areCheckpoints = true;

    if (time > 0) {
      // the restart_merger doesn't need checkpoints, and calls this
      // with time=0.
      copyTimesteps( checkpointsFromDir, d_checkpointsDir, startTimestep,
                     timestep, removeOldDir, areCheckpoints );
      copySection( checkpointsFromDir, d_checkpointsDir, "index.xml", "variables" );
      copySection( checkpointsFromDir, d_checkpointsDir, "index.xml", "globals" );
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
  else if( d_writeMeta ) { // Just add <restart from = ".." timestep = ".."> tag.
    copySection(restartFromDir, d_dir, "index.xml", "restarts");

    string iname = d_dir.getName()+"/index.xml";
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
DataArchiver::reduceUdaSetup(Dir& fromDir)
{
  //__________________________________
  // copy files
  // copy dat files and
  if (d_writeMeta) {
    d_fromDir = fromDir;
    copyDatFiles(fromDir, d_dir, 0, -1, false);
    copySection(fromDir,  d_dir, "index.xml", "globals");
    proc0cout << "*** Copied dat files to:   " << d_dir.getName() << "\n";
    
    // copy checkpoints
    Dir checkpointsFromDir = fromDir.getSubdir("checkpoints");
    Dir checkpointsToDir   = d_dir.getSubdir("checkpoints");
    string me = checkpointsFromDir.getName();
    if( validDir(me) ) {
      checkpointsToDir.remove( "index.xml", false);  // this file is created upstream when it shouldn't have
      checkpointsFromDir.copy( d_dir );
      proc0cout << "\n*** Copied checkpoints to: " << d_checkpointsDir.getName() << "\n";
      proc0cout << "    Only using 1 processor to copy so this will be slow for large checkpoint directories\n\n";
    }

    // copy input.xml.orig if it exists
    string there = d_dir.getName();
    string here  = fromDir.getName() + "/input.xml.orig";
    if ( validFile(here) ) {
      fromDir.copy("input.xml.orig", d_dir);     // use OS independent copy functions, needed by mira
      proc0cout << "*** Copied input.xml.orig to: " << there << "\n";
    }
    
    // copy the original ups file if it exists
    vector<string> ups;
    fromDir.getFilenamesBySuffix( "ups", ups );
    
    if ( ups.size() != 0 ) {
      fromDir.copy(ups[0], d_dir);              // use OS independent copy functions, needed by mira
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

      list<SaveNameItem>::iterator iter = d_saveLabelNames.begin();
      while ( iter != d_saveLabelNames.end() ) {
        if ( (*iter).labelName == varname ) {
          iter = d_saveLabelNames.erase(iter);
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
    d_restartTimestepIndicies[count] = timestep;
    
    ts = ts->findNextBlock("timestep");
    count ++;
  }
  
  d_restartTimestepIndicies[0] = d_restartTimestepIndicies[1];

  // Set checkpoint outputIntervals
  d_checkpointInterval = 0.0;
  d_checkpointTimestepInterval = 0;
  d_checkpointWalltimeInterval = 0;
  d_nextCheckpointTimestep  = SHRT_MAX;
  

  // output every timestep -- each timestep is transferring data
  d_outputInitTimestep     = true;
  d_outputInterval         = 0.0;
  d_outputTimestepInterval = 1;
  d_usingReduceUda         = true;

} // end reduceUdaSetup()

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
DataArchiver::copyTimesteps(Dir& fromDir, Dir& toDir, int startTimestep,
                            int maxTimestep, bool removeOld,
                            bool areCheckpoints /*=false*/)
{
   string old_iname = fromDir.getName()+"/index.xml";
   ProblemSpecP oldIndexDoc = loadDocument(old_iname);
   string iname = toDir.getName()+"/index.xml";
   ProblemSpecP indexDoc = loadDocument(iname);

   ProblemSpecP oldTimesteps = oldIndexDoc->findBlock("timesteps");

   ProblemSpecP ts;
   if( oldTimesteps != nullptr ) {
     ts = oldTimesteps->findBlock("timestep");
   }

   // while we're at it, add restart information to index.xml
   if( maxTimestep >= 0 ) {
     addRestartStamp(indexDoc, fromDir, maxTimestep);
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
      if (timestep >= startTimestep &&
          (timestep <= maxTimestep || maxTimestep < 0)) {
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
            d_checkpointTimestepDirs.push_back(toDir.getSubdir(href).getName());
         
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
DataArchiver::copyDatFiles(Dir& fromDir, Dir& toDir, int startTimestep,
                           int maxTimestep, bool removeOld)
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

         // copy up to maxTimestep lines of the old dat file to the copy
         int timestep = startTimestep;
         while (datFile.getline(buffer, 1000) &&
                (timestep < maxTimestep || maxTimestep < 0)) {
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

   rootElem->appendElement("numberOfProcessors", d_myworld->size());

   rootElem->appendElement("ParticlePosition", d_particlePositionName);
   
   string format = "uda";
   if ( d_outputFileFormat == PIDX ){
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
DataArchiver::finalizeTimestep( double        time, 
                                double        delt,
                                const GridP & grid, 
                                SchedulerP  & sched,
                                bool          recompile /* = false */ )
{
  //this function should get called exactly once per timestep
  
  //  static bool wereSavesAndCheckpointsInitialized = false;
  dbg << "  finalizeTimestep, time= " << time << " delt= " << delt << "\n";
  
  beginOutputTimestep( time, delt, grid );

  //__________________________________
  // some changes here - we need to redo this if we add a material, or
  // if we schedule output on the initialization timestep (because
  // there will be new computes on subsequent timestep) or if there is
  // a component switch or a new level in the grid - BJW
  if (((delt != 0.0 || d_outputInitTimestep) && !d_wereSavesAndCheckpointsInitialized) || 
      d_sharedState->getSwitchState() || grid->numLevels() != d_numLevelsInOutput) {
    // Skip the initialization timestep (normally, anyway) for this
    // because it needs all computes to be set to find the save labels    
    if( d_outputInterval > 0.0 || d_outputTimestepInterval > 0 ) {
      initSaveLabels(sched, delt == 0.0);
     
      if (!d_wereSavesAndCheckpointsInitialized && delt != 0.0) {
        indexAddGlobals(); // add saved global (reduction) variables to index.xml
      }
    }
    
    // This assumes that the TaskGraph doesn't change after the second
    // timestep and will need to change if the TaskGraph becomes dynamic. 
    // We also need to do this again if this is the initial timestep
    if (delt != 0.0) {
      d_wereSavesAndCheckpointsInitialized = true;
    
      // Can't do checkpoints on init timestep....
      if( d_checkpointInterval > 0.0 ||
	  d_checkpointTimestepInterval > 0 ||
	  d_checkpointWalltimeInterval > 0 ) {

        initCheckpoints( sched );
      }
    }
  }
  
  d_numLevelsInOutput = grid->numLevels();
  
#if SCI_ASSERTION_LEVEL >= 2
  d_outputCalled.clear();
  d_outputCalled.resize(d_numLevelsInOutput, false);
  d_checkpointCalled.clear();
  d_checkpointCalled.resize(d_numLevelsInOutput, false);
  d_checkpointReductionCalled = false;
#endif
}

//______________________________________________________________________
//  Schedule output tasks for the grid variables, particle variables and reduction variables
void
DataArchiver::sched_allOutputTasks(       double       delt,
                                    const GridP      & grid, 
                                          SchedulerP & sched,
                                          bool         recompile /* = false */ )
{
  dbg << "  sched_allOutputTasks \n";
  
  // we don't want to schedule more tasks unless we're recompiling
  if ( !recompile ) {
    return;
  }

  //__________________________________
  //  Reduction Variables
  // Schedule task to dump out reduction variables at every timestep
  
  if( (d_outputInterval  > 0.0 || d_outputTimestepInterval  > 0) &&
      (delt != 0.0 || d_outputInitTimestep)) {
    
    Task* task = scinew Task( "DataArchiver::outputReductionVars",
			   this, &DataArchiver::outputReductionVars );
    
    for( int i=0; i<(int)d_saveReductionLabels.size(); ++i) {
      SaveItem& saveItem = d_saveReductionLabels[i];
      const VarLabel* var = saveItem.label;
      
      const MaterialSubset* matls = saveItem.getMaterialSubset(0);
      task->requires( Task::NewDW, var, matls, true );
    }
    
    sched->addTask(task, nullptr, nullptr);
    
    dbg << "  scheduled output tasks (reduction variables)\n";

    if ( delt != 0.0 || d_outputInitTimestep ) {
      scheduleOutputTimestep( d_saveLabels, grid, sched, false );
    }
  }
  
  //__________________________________
  //  Schedule Checkpoint (reduction variables)
  if (delt != 0.0 && d_checkpointCycle > 0 &&
      ( d_checkpointInterval > 0 ||
	d_checkpointTimestepInterval > 0 ||
	d_checkpointWalltimeInterval > 0 ) ) {
    
    // output checkpoint timestep
    Task* task = scinew Task( "DataArchiver::outputVariables (CheckpointReduction)",
			   this, &DataArchiver::outputVariables, CHECKPOINT_REDUCTION );
    
    for( int i = 0; i < (int) d_checkpointReductionLabels.size(); i++ ) {
      SaveItem& saveItem = d_checkpointReductionLabels[i];
      const VarLabel* var = saveItem.label;
      const MaterialSubset* matls = saveItem.getMaterialSubset(0);
      
      task->requires(Task::NewDW, var, matls, true);
    }
    sched->addTask(task, nullptr, nullptr);
    
    dbg << "  scheduled output tasks (checkpoint variables)\n";
    
    scheduleOutputTimestep( d_checkpointLabels,  grid, sched, true );
  }
  
#if HAVE_PIDX
  if ( d_outputFileFormat == PIDX ) {
    /*  Create PIDX communicators (one communicator per AMR level) */
    // FIXME: It appears that this is called 3 or more times before timestep 0... why is this the case?
    //        Doesn't hurt anything, but we should figure out why...
    dbg << "  Creating communicatore per AMR level (required for PIDX)\n";
    createPIDXCommunicator( d_checkpointLabels,  grid, sched, true );
  }
#endif  

} // end sched_allOutputTasks()


//______________________________________________________________________
//
void
DataArchiver::beginOutputTimestep( double time, 
                                   double delt,
                                   const GridP& grid )
{
  // time should be currentTime+delt
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  dbg << "    beginOutputTimestep\n";

  // Do *not* update the next values here as the original values are
  // needed to compare with if there is a timestep restart.  See
  // reEvaluateOutputTimestep

  // Check for an output.
  d_isOutputTimestep =
    // Output based on the simulation time.
    ( ((d_outputInterval > 0.0 && (delt != 0.0 || d_outputInitTimestep)) &&
       (time + delt >= d_nextOutputTime) ) ||
			
      // Output based on the timestep interval.
      ((d_outputTimestepInterval > 0 && (delt != 0.0 || d_outputInitTimestep)) &&
       (timestep >= d_nextOutputTimestep)) ||

      // Output based on the being the last timestep.
      (d_outputLastTimestep && d_sharedState->maybeLast()) );

  // Create the output timestep directories
  if( d_isOutputTimestep && d_outputFileFormat != PIDX ) {
    makeTimestepDirs( d_dir, d_saveLabels, grid, &d_lastTimestepLocation );
  }

  
  // Check for a checkpoint.
  d_isCheckpointTimestep =
    // Checkpoint based on the simulation time.
    ( (d_checkpointInterval > 0.0 && (time + delt) >= d_nextCheckpointTime) ||
      
      // Checkpoint based on the timestep interval.
      (d_checkpointTimestepInterval > 0 && timestep >= d_nextCheckpointTimestep) ||

      // Checkpoint based on the being the last timestep.
      (d_checkpointLastTimestep && d_sharedState->maybeLast()) );    

  // Checkpoint based on the being the wall time.
  if( d_checkpointWalltimeInterval > 0 ) {

    // When using the wall time for checkpoints, rank 0 determines the
    // wall time and sends it to all other ranks.
    int walltime = d_sharedState->getElapsedWallTime();
    Uintah::MPI::Bcast( &walltime, 1, MPI_INT, 0, d_myworld->getComm() );

    if( walltime >= d_nextCheckpointWalltime )
      d_isCheckpointTimestep = true;	
  }
  
  // Create the output checkpoint directories
  if( d_isCheckpointTimestep ) {
    
    string timestepDir;
    makeTimestepDirs( d_checkpointsDir, d_checkpointLabels, grid, &timestepDir );
    
    string iname = d_checkpointsDir.getName() + "/index.xml";

    ProblemSpecP index;
    
    if (d_writeMeta) {
      index = loadDocument(iname);
      
      // store a back up in case it dies while writing index.xml
      string ibackup_name = d_checkpointsDir.getName()+"/index_backup.xml";
      index->output(ibackup_name.c_str());
    }

    d_checkpointTimestepDirs.push_back(timestepDir);
    
    if ((int)d_checkpointTimestepDirs.size() > d_checkpointCycle) {
      if (d_writeMeta) {
        // remove reference to outdated checkpoint directory from the
        // checkpoint index
        ProblemSpecP ts = index->findBlock("timesteps");
        ProblemSpecP temp = ts->getFirstChild();
        ts->removeChild(temp);

        index->output(iname.c_str());
        
        // remove out-dated checkpoint directory
        Dir expiredDir(d_checkpointTimestepDirs.front());

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
      d_checkpointTimestepDirs.pop_front();
    }
    //if (d_writeMeta)
    //index->releaseDocument();
  }

  dbg << "    write output timestep (" << d_isOutputTimestep << ")" << std::endl
      << "    write CheckPoints (" << d_isCheckpointTimestep << ")" << std::endl
      << "    end\n";
  
} // end beginOutputTimestep

//______________________________________________________________________
//
void
DataArchiver::makeTimestepDirs(       Dir                            & baseDir,
                                      vector<DataArchiver::SaveItem> & saveLabels ,
                                const GridP                          & grid,
                                      string                         * pTimestepDir /* passed back */ )
{
  int numLevels = grid->numLevels();
  // time should be currentTime+delt
  
  int timestep     = d_sharedState->getCurrentTopLevelTimeStep();
  int dir_timestep = getTimestepTopLevel();  // could be modified by reduceUda

  dbg << "      makeTimestepDirs for timestep: " << timestep << " dir_timestep: " << dir_timestep<< "\n";
  
  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << dir_timestep;
  *pTimestepDir = baseDir.getName() + "/" + tname.str();

  //__________________________________
  // Create the directory for this timestep, if necessary
  // It is not gurantteed that the rank holding d_writeMeta will call 
  // outputTimstep to create dir before another rank begin to output data.
  // A race condition happens when a rank executes output task and
  // the rank holding d_writeMeta is still compiling task graph. 
  // So every rank should try to create dir.
  //if(d_writeMeta) {

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
DataArchiver::reevaluate_OutputCheckPointTimestep( double time )
{
  dbg << "  reevaluate_OutputCheckPointTimestep() begin\n";

  // Call this on a timestep restart. If lowering the delt goes
  // beneath the threshold, cancel the output and/or checkpoint
  // timestep

  if (d_isOutputTimestep && d_outputInterval > 0.0 ) {
    if (time < d_nextOutputTime)
      d_isOutputTimestep = false;
  }
  
  if (d_isCheckpointTimestep && d_checkpointInterval > 0.0) {
    if (time < d_nextCheckpointTime) {
      d_isCheckpointTimestep = false;    
      d_checkpointTimestepDirs.pop_back();
    }
  }

#if SCI_ASSERTION_LEVEL >= 2
  d_outputCalled.clear();
  d_outputCalled.resize(d_numLevelsInOutput, false);
  d_checkpointCalled.clear();
  d_checkpointCalled.resize(d_numLevelsInOutput, false);
  d_checkpointReductionCalled = false;
#endif
  dbg << "  reevaluate_OutputCheckPointTimestep() end\n";
}

//______________________________________________________________________
//
void
DataArchiver::findNext_OutputCheckPointTimestep( double time, bool restart )
{
  dbg << "  findNext_OutputCheckPoint_Timestep() begin\n";

  int timestep = d_sharedState->getCurrentTopLevelTimeStep();

  if( restart )
  {
    // Output based on the simulaiton time.
    if( d_outputInterval > 0.0 ) {
      d_nextOutputTime = ceil(time / d_outputInterval) * d_outputInterval;
    }
    // Output based on the time step.
    else if( d_outputTimestepInterval > 0 ) {
      d_nextOutputTimestep =
	(timestep/d_outputTimestepInterval) * d_outputTimestepInterval + 1;

      while( d_nextOutputTimestep <= timestep ) {
	d_nextOutputTimestep += d_outputTimestepInterval;
      }
    }
   
    // Checkpoint based on the simulaiton time.
    if( d_checkpointInterval > 0.0 ) {
      d_nextCheckpointTime =
	ceil(time / d_checkpointInterval) * d_checkpointInterval;
    }
    // Checkpoint based on the time step.
    else if( d_checkpointTimestepInterval > 0 ) {
      d_nextCheckpointTimestep =
	(timestep / d_checkpointTimestepInterval) *
	d_checkpointTimestepInterval + 1;
      while( d_nextCheckpointTimestep <= timestep ) {
	d_nextCheckpointTimestep += d_checkpointTimestepInterval;
      }
    }
    // Checkpoint based on the wall time.
    else if( d_checkpointWalltimeInterval > 0 ) {
      // When using the wall time for checkpoints, rank 0 determines the
      // wall time and sends it to all other ranks.
      int walltime = d_sharedState->getElapsedWallTime();
      Uintah::MPI::Bcast( &walltime, 1, MPI_INT, 0, d_myworld->getComm() );

      d_nextCheckpointWalltime = walltime + d_checkpointWalltimeInterval;
    }
  }
  
  // If this timestep was an output/checkpoint timestep, determine
  // when the next one will be.

  // Do *not* do this step in beginOutputTimestep because the original
  // values are needed to compare with if there is a timestep restart.
  // See reEvaluateOutputTimestep

  // When outputing/checkpointing using the simulation or wall time
  // check to see if the simulation or wall time went past more than
  // one interval. If so adjust accordingly.

  // Note - it is not clear why but when outputing/checkpointing using
  // time steps the mod function must also be used. This does not
  // affect most simulations except when there are multiple UPS files
  // such as when components are switched. For example:
  // StandAlone/inputs/UCF/Switcher/switchExample3.ups

  else if( d_isOutputTimestep ) {
    
    // Output based on the simulaiton time.
    if( d_outputInterval > 0.0 ) {
      if( time >= d_nextOutputTime ) {
        d_nextOutputTime +=
	  floor( (time - d_nextOutputTime) / d_outputInterval ) *
	  d_outputInterval + d_outputInterval;
      }
    }
    // Output based on the time step.
    else if( d_outputTimestepInterval > 0 ) {
      if( timestep >= d_nextOutputTimestep )  {
        d_nextOutputTimestep +=
	  ( (timestep-d_nextOutputTimestep) / d_outputTimestepInterval ) *
	  d_outputTimestepInterval + d_outputTimestepInterval;
      }
    }
  }

  if( d_isCheckpointTimestep ) {
    // Checkpoint based on the simulaiton time.
    if( d_checkpointInterval > 0.0 ) {
      if( time >= d_nextCheckpointTime ) {
        d_nextCheckpointTime +=
	  floor( (time - d_nextCheckpointTime) / d_checkpointInterval ) *
	  d_checkpointInterval + d_checkpointInterval;
      }
    }
    // Checkpoint based on the time step.
    else if( d_checkpointTimestepInterval > 0 ) {
      if( timestep >= d_nextCheckpointTimestep ) {
        d_nextCheckpointTimestep +=
	  ( (timestep - d_nextCheckpointTimestep) /
	    d_checkpointTimestepInterval ) *
	  d_checkpointTimestepInterval + d_checkpointTimestepInterval;
      }
    }

    // Checkpoint based on the wall time.
    else if( d_checkpointWalltimeInterval > 0 ) {

      // When using the wall time for checkpoints, rank 0 determines
      // the wall time and sends it to all other ranks.
      int walltime = d_sharedState->getElapsedWallTime();
      Uintah::MPI::Bcast( &walltime, 1, MPI_INT, 0, d_myworld->getComm() );

      if( walltime >= d_nextCheckpointWalltime ) {
        d_nextCheckpointWalltime +=
	  floor( (walltime - d_nextCheckpointWalltime) /
		 d_checkpointWalltimeInterval ) *
	  d_checkpointWalltimeInterval + d_checkpointWalltimeInterval;
      }
    }
  }  
  
  dbg << "    next output sim time: " << d_nextOutputTime 
      << "  next output Timestep: " << d_nextOutputTimestep << "\n"
      << "    next checkpoint sim time: " << d_nextCheckpointTime 
      << "  next checkpoint timestep: " << d_nextCheckpointTimestep
      << "  next checkpoint walltime: " << d_nextCheckpointWalltime << "\n";

  dbg << "  findNext_OutputCheckPoint_Timestep() end\n";

} // end findNext_OutputCheckPoint_Timestep()


//______________________________________________________________________
//  update the xml files (index.xml, timestep.xml, 
void
DataArchiver::writeto_xml_files( double delt, const GridP& grid )
{
  Timers::Simple timer;
  timer.start();
  
  dbg << "  writeto_xml_files() begin\n";

  if( !d_isCheckpointTimestep && !d_isOutputTimestep ) {
    dbg << "   This is not an output (or checkpoint) timestep, so just returning...\n";
    return;
  }
  
  //__________________________________
  //  Writeto XML files
  // to check for output nth proc
  int dir_timestep = getTimestepTopLevel();  // could be modified by reduceUda
  
  // start dumping files to disk
  vector<Dir*> baseDirs;
  if ( d_isOutputTimestep ) {
    baseDirs.push_back( &d_dir );
  }    
  if ( d_isCheckpointTimestep ) {
    baseDirs.push_back( &d_checkpointsDir );
  }

  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << dir_timestep;

  for (int i = 0; i < static_cast<int>( baseDirs.size() ); ++i) {
    // to save the list of vars. up to 2, since in checkpoints, there
    // are two types of vars
    vector< vector<SaveItem>* > savelist; 
    
    // Reference this timestep in index.xml
    if(d_writeMeta) {
      bool hasGlobals = false;

      if ( baseDirs[i] == &d_dir ) {
        savelist.push_back( &d_saveLabels );
      }
      else if ( baseDirs[i] == &d_checkpointsDir ) {
        hasGlobals = d_checkpointReductionLabels.size() > 0;
        savelist.push_back( &d_checkpointLabels );
        savelist.push_back( &d_checkpointReductionLabels );
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
        string variableSection = savelist[j] == &d_checkpointReductionLabels ? "globals" : "variables";
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
                                                                 baseDirs[i] != &d_dir ) );
            newElem->setAttribute("name", var->getName());
          }
        }
      }

      //__________________________________
      // Check if it's the first checkpoint timestep by checking if
      // the "timesteps" field is in checkpoints/index.xml.  If it is
      // then there exists a timestep.xml file already.  Use this
      // below to change information in input.xml...
      bool firstCheckpointTimestep = false;
      
      ProblemSpecP ts = indexDoc->findBlock("timesteps");
      if( ts == nullptr ) {
        ts = indexDoc->appendChild("timesteps");
        firstCheckpointTimestep = (&d_checkpointsDir == baseDirs[i]);
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

      double time = d_sharedState->getElapsedSimTime();
      
      //__________________________________
      // add timestep info
      if(!found) {
        
        string timestepindex = tname.str()+"/timestep.xml";      
        
        ostringstream value, timeVal, deltVal;
        value << dir_timestep;
        ProblemSpecP newElem = ts->appendElement( "timestep",value.str().c_str() );
        newElem->setAttribute( "href",     timestepindex.c_str() );
        timeVal << std::setprecision(17) << time + delt;
        newElem->setAttribute( "time",     timeVal.str() );
        deltVal << std::setprecision(17) << delt;
        newElem->setAttribute( "oldDelt",  deltVal.str() );
      }
      
      indexDoc->output(iname.c_str());
      //indexDoc->releaseDocument();

      // make a timestep.xml file for this timestep we need to do it
      // here in case there is a timestesp restart Break out the
      // <Grid> and <Data> section of the DOM tree into a separate
      // grid.xml file which can be created quickly and use less
      // memory using the xmlTextWriter functions (streaming output)

      ProblemSpecP rootElem = ProblemSpec::createDocument( "Uintah_timestep" );

      // Create a metadata element to store the per-timestep endianness
      ProblemSpecP metaElem = rootElem->appendChild("Meta");

      metaElem->appendElement("endianness", endianness().c_str());
      metaElem->appendElement("nBits", (int)sizeof(unsigned long) * 8 );
      metaElem->appendElement("numProcs", d_myworld->size());

      // Timestep information
      ProblemSpecP timeElem = rootElem->appendChild("Time");
      timeElem->appendElement("timestepNumber", dir_timestep);
      timeElem->appendElement("currentTime", time + delt);
      timeElem->appendElement("oldDelt", delt);

      //__________________________________
      // Output grid section:
      //
      // With AMR, we're not guaranteed that a rank has work on a
      // given level.  Quick check to see that, so we don't create a
      // node that points to no data.

      string grid_path = baseDirs[i]->getName() + "/" + tname.str() + "/";

#if XML_TEXTWRITER

      writeGridTextWriter( hasGlobals, grid_path, grid );
#else
      // Original version:
      writeGridOriginal( hasGlobals, grid, rootElem );

      // Binary Grid version:
      // writeGridBinary( hasGlobals, grid_path, grid );
#endif
      // Add the <Materials> section to the timestep.xml
      SimulationInterface* sim = dynamic_cast<SimulationInterface*>(getPort("sim")); 

      GeometryPieceFactory::resetGeometryPiecesOutput();

      // output each components output Problem spec
      sim->outputProblemSpec( rootElem );

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
      
      if (firstCheckpointTimestep) {
        // loop over the blocks in timestep.xml and remove them from
        // input.xml, with some exceptions.
        string inputname = d_dir.getName()+"/input.xml";
        ProblemSpecP inputDoc = loadDocument(inputname);
        inputDoc->output((inputname + ".orig").c_str());

        for (ProblemSpecP ps = rootElem->getFirstChild(); ps != nullptr; ps = ps->getNextSibling()) {
          string nodeName = ps->getNodeName();
          
          if (nodeName == "Meta" || nodeName == "Time" || nodeName == "Grid" || nodeName == "Data") {
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
      if( d_usingReduceUda ) {
        copy_outputProblemSpec( d_fromDir, d_dir );
      }
    }
  }  // loop over baseDirs

  double myTime = timer().seconds();
  d_sharedState->d_runTimeStats[SimulationState::XMLIOTime] += myTime;
  d_sharedState->d_runTimeStats[SimulationState::TotalIOTime ] += myTime;

  dbg << "  end\n";
}

//______________________________________________________________________
//  update the xml file index.xml with any in-situ modified variables.
void
DataArchiver::writeto_xml_files( std::map< std::string,
				 std::pair<std::string,
				 std::string> > &modifiedVars )
{
#ifdef HAVE_VISIT  
  if( isProc0_macro && d_sharedState->getVisIt() && modifiedVars.size() )
  {
    dbg << "  writeto_xml_files() begin\n";

    //__________________________________
    //  Writeto XML files
    // to check for output nth proc
    int dir_timestep = getTimestepTopLevel(); // could be modified by reduceUda
  
    string iname = d_dir.getName()+"/index.xml";

    ProblemSpecP indexDoc = loadDocument(iname);
    
    string inSituSection("InSitu");
	  
    ProblemSpecP iss = indexDoc->findBlock(inSituSection);
	  
    if(iss.get_rep() == nullptr) {
      iss = indexDoc->appendChild(inSituSection.c_str());
    }
	  
    std::stringstream timestep;
    timestep << d_sharedState->getCurrentTopLevelTimeStep()+1;
      
    // Report on the modiied variables. 
    std::map<std::string,std::pair<std::string, std::string> >::iterator iter;
    for (iter = modifiedVars.begin(); iter != modifiedVars.end(); ++iter) {
      proc0cout << "Visit libsim - For time step " << timestep.str() << " "
		<< "the variable "         << iter->first << " "
		<< "will be changed from " << iter->second.first << " "
		<< "to "                   << iter->second.second << ". "
		<< std::endl;

      ProblemSpecP newElem = iss->appendChild("modifiedVariable");

      newElem->setAttribute("timestep", timestep.str());
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
DataArchiver::writeGridBinary( const bool hasGlobals, const string & grid_path, const GridP & grid )
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

  string grid_filename = grid_path + "grid.xml";

  fp = fopen( grid_filename.c_str(), "wb" );
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

    LoadBalancerPort * lb = dynamic_cast<LoadBalancerPort*>( getPort("load balancer") );

    procOnLevel[ lev ].resize( d_myworld->size() );

    // Iterate over patches.
    Level::const_patch_iterator iter;
    for( iter = level->patchesBegin(); iter != level->patchesEnd(); ++iter ) {
      const Patch* patch = *iter;

      int       patch_id   = patch->getID();
      int       rank_id    = lb->getOutputRank( patch );

      int proc = lb->getOutputRank( patch );
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

  writeDataTextWriter( hasGlobals, grid_path, grid, procOnLevel );

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

  LoadBalancerPort * lb = dynamic_cast<LoadBalancerPort*>(getPort("load balancer"));

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

    procOnLevel[ l ].resize( d_myworld->size() );

    Level::const_patch_iterator iter;
    for( iter = level->patchesBegin(); iter != level->patchesEnd(); ++iter ) {
      const Patch* patch = *iter;
          
      IntVector lo = patch->getCellLowIndex();    // for readability
      IntVector hi = patch->getCellHighIndex();
      IntVector lo_EC = patch->getExtraCellLowIndex();
      IntVector hi_EC = patch->getExtraCellHighIndex();
          
      int proc = lb->getOutputRank( patch );
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

    for( int i = 0; i < d_myworld->size(); i++ ) {
      if( ( i % lb->getNthRank() ) != 0 || !procOnLevel[l][i] ){
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
DataArchiver::writeGridTextWriter( const bool hasGlobals, const string & grid_path, const GridP & grid )
{
  // With AMR, we're not guaranteed that a proc do work on a given
  // level.  Quick check to see that, so we don't create a node that
  // points to no data
  int numLevels = grid->numLevels();
  vector< vector<bool> > procOnLevel( numLevels );

  // Break out the <Grid> and <Data> sections and write those to
  // grid.xml and data.xml files using libxml2's TextWriter which is a
  // streaming output format which doesn't use a DOM tree.

  LoadBalancerPort * lb =
    dynamic_cast<LoadBalancerPort*>( getPort("load balancer") );

  string name_grid = grid_path + "grid.xml";

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
    procOnLevel[ l ].resize( d_myworld->size() );

    Level::const_patch_iterator iter;
    for(iter=level->patchesBegin(); iter != level->patchesEnd(); ++iter) {
      const Patch* patch = *iter;
          
      IntVector lo = patch->getCellLowIndex();    // for readability
      IntVector hi = patch->getCellHighIndex();
      IntVector lo_EC = patch->getExtraCellLowIndex();
      IntVector hi_EC = patch->getExtraCellHighIndex();
          
      int proc = lb->getOutputRank( patch );
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

  writeDataTextWriter( hasGlobals, grid_path, grid, procOnLevel );

} // end writeGridTextWriter()

void
DataArchiver::writeDataTextWriter( const bool hasGlobals, const string & data_path, const GridP & grid,
                                   const vector< vector<bool> > & procOnLevel )
{
  int                   numLevels = grid->numLevels();
  LoadBalancerPort    * lb = dynamic_cast<LoadBalancerPort*>( getPort("load balancer") );

  string                data_filename = data_path + "data.xml";
  xmlTextWriterPtr      data_writer = xmlNewTextWriterFilename( data_filename.c_str(), 0 );

  xmlTextWriterSetIndent( data_writer, 2 );
  xmlTextWriterStartElement( data_writer, BAD_CAST "Data" );

  for( int l = 0; l < numLevels; l++ ) {
    ostringstream lname;
    lname << "l" << l;

    // create a pxxxxx.xml file for each proc doing the outputting
    for( int i = 0; i < d_myworld->size(); i++ ) {
      if ( i % lb->getNthRank() != 0 || !procOnLevel[l][i] ) {
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
DataArchiver::scheduleOutputTimestep(       vector<SaveItem> & saveLabels,
                                      const GridP            & grid, 
                                            SchedulerP       & sched,
                                            bool               isThisCheckpoint )
{
  // Schedule a bunch o tasks - one for each variable, for each patch
  int                var_cnt = 0;
  LoadBalancerPort * lb =
    dynamic_cast< LoadBalancerPort * >( getPort( "load balancer" ) ); 
  
  for( int i = 0; i < grid->numLevels(); i++ ) {

    const LevelP& level = grid->getLevel(i);
    const PatchSet* patches = lb->getOutputPerProcessorPatchSet( level );
    
    string taskName = "DataArchiver::outputVariables";
    if ( isThisCheckpoint ) {
      taskName += "(checkpoint)";
    }
    
    Task* t = scinew Task( taskName, this, &DataArchiver::outputVariables,
			   isThisCheckpoint ? CHECKPOINT : OUTPUT );
    
    //__________________________________
    //
    vector< SaveItem >::iterator saveIter;
    for( saveIter = saveLabels.begin();
	 saveIter != saveLabels.end(); ++saveIter ) {
      const MaterialSubset* matls = saveIter->getMaterialSubset(level.get_rep());
      
      if ( matls != nullptr ) {
        t->requires( Task::NewDW, (*saveIter).label, matls, Task::OutOfDomain, Ghost::None, 0, true );
        var_cnt++;
      }
    }

    t->setType( Task::Output );
    sched->addTask( t, patches, d_sharedState->allMaterials() );
  }
  
  dbg << "  scheduled output task for " << var_cnt << " variables\n";
}

#if HAVE_PIDX
void
DataArchiver::createPIDXCommunicator(       vector<SaveItem> & saveLabels,
                                      const GridP            & grid, 
                                            SchedulerP       & sched,
                                            bool               isThisCheckpoint )
{
  int proc = d_myworld->myrank();
  LoadBalancerPort * lb = dynamic_cast< LoadBalancerPort * >( getPort( "load balancer" ) );

  // Resize the comms back to 0...
  d_pidxComms.clear();

  // Create new MPI Comms
  d_pidxComms.reserve( grid->numLevels() );
  
  for( int i = 0; i < grid->numLevels(); i++ ) {

    const LevelP& level = grid->getLevel(i);
    vector< SaveItem >::iterator saveIter;
    const PatchSet* patches = lb->getOutputPerProcessorPatchSet( level );
    //cout << "[ "<< d_myworld->myrank() << " ] Patch size: " << patches->size() << "\n";
    
    /*
      int color = 0;
      if (patches[d_myworld->myrank()].size() != 0)
        color = 1;
      MPI_Comm_split(d_myworld->getComm(), color, d_myworld->myrank(), &(pidxComms[i]));
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
    
    MPI_Comm_split( d_myworld->getComm(), color, d_myworld->myrank(), &(d_pidxComms[i]) );
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
    return d_dir.getName();
}

//______________________________________________________________________
//
void
DataArchiver::indexAddGlobals()
{
  dbg << "  indexAddGlobals()\n";

  // add info to index.xml about each global (reduction) var assume
  // for now that global variables that get computed will not change
  // from timestep to timestep
  static bool wereGlobalsAdded = false;
  if (d_writeMeta && !wereGlobalsAdded) {
    wereGlobalsAdded = true;
    // add saved global (reduction) variables to index.xml
    string iname = d_dir.getName()+"/index.xml";
    ProblemSpecP indexDoc = loadDocument(iname);
    
    ProblemSpecP globals = indexDoc->appendChild("globals");

    vector< SaveItem >::iterator saveIter;
    for (saveIter = d_saveReductionLabels.begin();
	 saveIter != d_saveReductionLabels.end(); ++saveIter) {
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
  if( new_dw->timestepRestarted() || d_saveReductionLabels.empty() ) {
    return;
  }
    
  dbg << "  outputReductionVars task begin\n";

  Timers::Simple timer;
  timer.start();

  double time = d_sharedState->getElapsedSimTime();

  delt_vartype delt_var(0);
  if( old_dw )
    old_dw->get( delt_var, d_sharedState->get_delt_label() );
  double delt = delt_var;

  // Dump the stuff in the reduction saveset into files in the uda
  // at every timestep
  for(int i=0; i<(int)d_saveReductionLabels.size(); ++i) {
    SaveItem& saveItem = d_saveReductionLabels[i];
    const VarLabel* var = saveItem.label;
    // FIX, see above
    const MaterialSubset* matls =
      saveItem.getMaterialSet(ALL_LEVELS)->getUnion();
    
    for (int m = 0; m < matls->size(); m++) {
      int matlIndex = matls->get(m);
      dbg << "    Reduction " << var->getName() << " matl: " << matlIndex << "\n";
      ostringstream filename;
      filename << d_dir.getName() << "/" << var->getName();
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
      
      out << std::setprecision(17) << time + delt << "\t";
      new_dw->print(out, var, 0, matlIndex);
      out << "\n";
    }
  }

  double myTime = timer().seconds();
  d_sharedState->d_runTimeStats[SimulationState::ReductionIOTime] += myTime;
  d_sharedState->d_runTimeStats[SimulationState::TotalIOTime ] += myTime;
  
  dbg << "  outputReductionVars task end\n";
}

//______________________________________________________________________
//
void
DataArchiver::outputVariables( const ProcessorGroup * pg,
                               const PatchSubset    * patches,
                               const MaterialSubset * /*matls*/,
                               DataWarehouse        * /*old_dw*/,
                               DataWarehouse        * new_dw,
                               int                    type )
{
  // IMPORTANT - this function should only be called once per
  //   processor per level per type (files will be opened and closed,
  //   and those operations are heavy on parallel file systems)

  // return if not an outpoint/checkpoint timestep
  if ((!d_isOutputTimestep && type == OUTPUT) || 
      (!d_isCheckpointTimestep &&
       (type == CHECKPOINT || type == CHECKPOINT_REDUCTION))) {
    return;
  }

  dbg << "  outputVariables task begin\n";

#if SCI_ASSERTION_LEVEL >= 2
  // double-check to make sure only called once per level
  int levelid =
    type != CHECKPOINT_REDUCTION ? getLevel(patches)->getIndex() : -1;
  
  if (type == OUTPUT) {
    ASSERT(d_outputCalled[levelid] == false);
    d_outputCalled[levelid] = true;
  }
  else if (type == CHECKPOINT) {
    ASSERT(d_checkpointCalled[levelid] == false);
    d_checkpointCalled[levelid] = true;
  }
  else /* if (type == CHECKPOINT_REDUCTION) */ {
    ASSERT(d_checkpointReductionCalled == false);
    d_checkpointReductionCalled = true;
  }
#endif

  vector< SaveItem >& saveLabels =
	 (type == OUTPUT ? d_saveLabels :
	  type == CHECKPOINT ? d_checkpointLabels : d_checkpointReductionLabels);

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
    dbg << " on timestep: " << d_sharedState->getCurrentTopLevelTimeStep() << "\n";
  }
    
  
  //__________________________________
  Dir dir;
  if (type == OUTPUT) {
    dir = d_dir;
  }
  else /* if (type == CHECKPOINT || type == CHECKPOINT_REDUCTION) */ {
    dir = d_checkpointsDir;
  }
  
  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << getTimestepTopLevel();    // could be modified by reduceUda

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
    pname << "p" << setw(5) << setfill('0') << d_myworld->myrank();
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

  if (d_outputFileFormat == UDA || type == CHECKPOINT_REDUCTION)
  {  
    d_outputLock.lock(); 
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
        dbg << "    "<<var->getName() << ", materials: ";
        for(int m=0;m<var_matls->size();m++) {
          if(m != 0)
            dbg << ", ";
          dbg << var_matls->get(m);
        }
        dbg <<"\n";
        
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
            OutputContext oc(fd, filename, cur, pdElem, d_outputDoubleAsFloat && type != CHECKPOINT);
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

    d_outputLock.unlock(); 
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
  if ( d_outputFileFormat == PIDX && type != CHECKPOINT_REDUCTION ) {
  
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
      vector<SaveItem> saveTheseLabels;
      saveTheseLabels = findAllVariableTypes( saveLabels, TD );
      
      if( saveTheseLabels.size() > 0 ) {
        string dirName = pidx.getDirectoryName( TD );

        Dir myDir = ldir.getSubdir( dirName );
        
        totalBytes += saveLabels_PIDX(saveTheseLabels, pg, patches,
				      new_dw, type, TD, ldir, dirName, doc);
      } 
    }

    // write the xml 
    //doc->output(xmlFilename.c_str());
  }
#endif

  double myTime = timer().seconds();
  double byteToMB = 1024*1024;

  if (type == OUTPUT) {
    d_sharedState->d_runTimeStats[SimulationState::OutputIOTime] +=
      myTime;
    d_sharedState->d_runTimeStats[SimulationState::OutputIORate] +=
      (double) totalBytes / (byteToMB * myTime);
  }
  else if (type == CHECKPOINT ) {
    d_sharedState->d_runTimeStats[SimulationState::CheckpointIOTime] +=
      myTime;
    d_sharedState->d_runTimeStats[SimulationState::CheckpointIORate] +=
      (double) totalBytes / (byteToMB * myTime);
  }
    
  else /* if (type == CHECKPOINT_REDUCTION) */ {
    d_sharedState->d_runTimeStats[SimulationState::CheckpointReductionIOTime] +=
      myTime;
    d_sharedState->d_runTimeStats[SimulationState::CheckpointReducIORate] +=
      (double) totalBytes / (byteToMB * myTime);
  }
    
  d_sharedState->d_runTimeStats[SimulationState::TotalIOTime ] += myTime;

  dbg << "  outputVariables task end\n";
} // end outputVariables()

//______________________________________________________________________
//  output only the savedLabels of a specified type description in PIDX format.

size_t
DataArchiver::saveLabels_PIDX( std::vector< SaveItem >     & saveLabels,
                               const ProcessorGroup        * pg,
                               const PatchSubset           * patches,      
                               DataWarehouse               * new_dw,          
                               int                           type,
                               const TypeDescription::Type   TD,
                               Dir                           ldir,        // uda/timestep/levelIndex
                               const std::string           & dirName,     // CCVars, SFC*Vars
                               ProblemSpecP                & doc )
{
  size_t totalBytesSaved = 0;
#if HAVE_PIDX
  int levelid = getLevel(patches)->getIndex(); 
  const Level* level = getLevel(patches);

  int nSaveItems =  saveLabels.size();
  vector<int> nSaveItemMatls (nSaveItems);

  int rank = pg->myrank();
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

  pidx.setOutputDoubleAsFloat( (d_outputDoubleAsFloat && type == OUTPUT) );  

  unsigned int timeStep = d_sharedState->getCurrentTopLevelTimeStep();
  
  //__________________________________
  // define the level extents for this variable type
  IntVector lo;
  IntVector hi;
  level->computeVariableExtents(TD,lo, hi);
  
  PIDX_point level_size;
  pidx.setLevelExtents( "DataArchiver::saveLabels_PIDX",  lo, hi, level_size );

  // Can this be run in serial without doing a MPI initialize
  pidx.initialize( full_idxFilename, timeStep, /*d_myworld->getComm()*/d_pidxComms[ levelid ], d_PIDX_flags, patches, level_size, type );

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

      rc = PIDX_append_and_write_variable(pidx.file, pidx.varDesc[vc][m]);
      pidx.checkReturnCode( rc,
			    "DataArchiver::saveLabels_PIDX - PIDX_append_and_write_variable failure",
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
    vector<TypeDescription::Type>::iterator td_iter;
    for (td_iter = pidxVarTypes.begin(); td_iter!= pidxVarTypes.end(); ++td_iter) {
      TypeDescription::Type TD = *td_iter;
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

  if( d_outputFileFormat == UDA ) {
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

  if (d_udaSuffix != -1) {
    ostringstream name;
    name << d_filebase << "." << setw(3) << setfill('0') << d_udaSuffix;
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
    name << d_filebase << "." << setw(3) << setfill('0') << dirNum;
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
  int rc = LSTAT(d_filebase.c_str(), &sb);
  if ((rc != 0) && (errno == ENOENT))
    make_link = true;
  else if ((rc == 0) && (S_ISLNK(sb.st_mode))) {
    unlink(d_filebase.c_str());
    make_link = true;
  }
  if (make_link)
    symlink(dirName.c_str(), d_filebase.c_str());

  cout << "DataArchiver created " << dirName << "\n";
  d_dir = Dir(dirName);
   
} // end makeVersionedDir()


//______________________________________________________________________
//  Determine which labels will be saved.
void
DataArchiver::initSaveLabels(SchedulerP& sched, bool initTimestep)
{
  dbg << "  initSaveLabels()\n";

  // if this is the initTimestep, then don't complain about saving all
  // the vars, just save the ones you can.  They'll most likely be
  // around on the next timestep.
 
  SaveItem saveItem;
  d_saveReductionLabels.clear();
  d_saveLabels.clear();
   
  d_saveLabels.reserve( d_saveLabelNames.size() );
  Scheduler::VarLabelMaterialMap* pLabelMatlMap;
  pLabelMatlMap = sched->makeVarLabelMaterialMap();

  // iterate through each of the saveLabelNames we created in problemSetup
  list<SaveNameItem>::iterator iter;
  for (iter = d_saveLabelNames.begin();
       iter != d_saveLabelNames.end(); ++iter) {
    VarLabel* var = VarLabel::find((*iter).labelName);
    
    //   see if that variable has been created, set the compression
    //   mode make sure that the scheduler shows that that it has been
    //   scheduled to be computed.  Then save it to saveItems.
    if (var == nullptr) {
      if (initTimestep) {
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
      if (initTimestep) {
        // ignore this on the init timestep, cuz lots of vars aren't
        // computed on the init timestep
        dbg << "    Ignoring var " << iter->labelName
	    << " on initialization timestep\n";
        continue;
      } else {
        throw ProblemSetupException((*iter).labelName +
				    " variable not computed for saving.",
				    __FILE__, __LINE__);
      }
    }
    saveItem.label = var;
    saveItem.matlSet.clear();
    
    for (ConsecutiveRangeSet::iterator crs_iter = (*iter).levels.begin();
	 crs_iter != (*iter).levels.end(); ++crs_iter) {

      ConsecutiveRangeSet matlsToSave = (ConsecutiveRangeSet((*found).second)).intersected((*iter).matls);
      saveItem.setMaterials(*crs_iter, matlsToSave, d_prevMatls, d_prevMatlSet);

      if (((*iter).matls != ConsecutiveRangeSet::all) && ((*iter).matls != matlsToSave)) {
        throw ProblemSetupException((*iter).labelName +
				    " variable not computed for all materials specified to save.",
				    __FILE__, __LINE__);
      }
    }
    
    if (saveItem.label->typeDescription()->isReductionVariable()) {
      d_saveReductionLabels.push_back(saveItem);
    } else {
      d_saveLabels.push_back(saveItem);
    }
  }
  
  //d_saveLabelNames.clear();
  delete pLabelMatlMap;
}


//______________________________________________________________________
//
void
DataArchiver::initCheckpoints(SchedulerP& sched)
{
   dbg << "  initCheckpoints()\n";
   typedef vector<const Task::Dependency*> dep_vector;
   const dep_vector& initreqs = sched->getInitialRequires();
   
   // special variables to not checkpoint
   const set<string>& notCheckPointVars = sched->getNotCheckPointVars();
   
   SaveItem saveItem;
   d_checkpointReductionLabels.clear();
   d_checkpointLabels.clear();

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
         
   d_checkpointLabels.reserve(label_map.size());
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
       
       saveItem.setMaterials(map_iter->first, map_iter->second, d_prevMatls, d_prevMatlSet);

       if (string(var->getName()) == "delT") {
         hasDelT = true;
       }
     }
     
     // Skip this variable if the default behavior of variable has
     // been overwritten.  For example ignore checkpointing
     // PerPatch<FileInfo> variable
     bool skipVar = ( notCheckPointVars.count(saveItem.label->getName() ) > 0 );
     
     if( !skipVar ) {
       if ( saveItem.label->typeDescription()->isReductionVariable() ) {
         d_checkpointReductionLabels.push_back(saveItem);
       } else {
         d_checkpointLabels.push_back(saveItem);
       }
     }
   }


   if (!hasDelT) {
     VarLabel* var = VarLabel::find("delT");
     if (var == nullptr) {
       throw ProblemSetupException("delT variable not found to checkpoint.",__FILE__, __LINE__);
     }
     
     saveItem.label = var;
     saveItem.matlSet.clear();
     ConsecutiveRangeSet globalMatl("-1");
     saveItem.setMaterials(-1,globalMatl, d_prevMatls, d_prevMatlSet);
     ASSERT(saveItem.label->typeDescription()->isReductionVariable());
     d_checkpointReductionLabels.push_back(saveItem);
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
DataArchiver::needRecompile(double /*time*/, double /*dt*/,
                            const GridP& /*grid*/)
{
  return false;
/*
  LevelP level = grid->getLevel(0);
  d_currentTime=time+dt;
  dbg << "DataArchiver::needRecompile called\n";
  d_currentTimestep++;
  bool recompile=false;
  bool do_output=false;
  if ((d_outputInterval > 0.0 && time+dt >= d_nextOutputTime) ||
      (d_outputTimestepInterval > 0 &&
      d_currentTimestep+1 > d_nextOutputTimestep)) {
    do_output=true;
    if(!d_wasOutputTimestep)
      recompile=true;
  } else {
    if(d_wasOutputTimestep)
      recompile=true;
  }

  // When using the wall clock time for checkpoints, rank 0 determines
  // the wall time and sends it to all other ranks.
  int walltime = d_sharedState->getElapsedWallTime();
  Uintah::MPI::Bcast( &walltime, 1, MPI_INT, 0, d_myworld->getComm() );

  if ((d_checkpointInterval > 0.0 && time+dt >= d_nextCheckpointTime) ||
      (d_checkpointTimestepInterval > 0 &&
      d_currentTimestep+1 > d_nextCheckpointTimestep) ||
      (d_checkpointWalltimeInterval > 0 &&
      walltime >= d_nextCheckpointWalltime)) {
    do_output=true;
    if(!d_wasCheckpointTimestep)
      recompile=true;
  } else {
    if(d_wasCheckpointTimestep)
      recompile=true;
  }
  dbg << "wasOutputTimestep=" << d_wasOutputTimestep << '\n';
  dbg << "wasCheckpointTimestep=" << d_wasCheckpointTimestep << '\n';
  dbg << "time=" << time << '\n';
  dbg << "dt=" << dt << '\n';
  dbg << "do_output=" << do_output << '\n';
  if(do_output)
    beginOutputTimestep(time, dt, grid);
  else
    d_wasCheckpointTimestep=d_wasOutputTimestep=false;
  if(recompile)
    dbg << "We do request recompile\n";
  else
    dbg << "We do not request recompile\n";

  return recompile;
*/
}

//______________________________________________________________________
//
string
DataArchiver::TranslateVariableType( string type, bool isThisCheckpoint )
{
  if ( d_outputDoubleAsFloat && !isThisCheckpoint ) {
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
  if(d_outputInterval == 0.0 && d_outputTimestepInterval == 0) {
    return false;
  }

  list<SaveNameItem>::const_iterator iter;
  for( iter = d_saveLabelNames.begin(); iter != d_saveLabelNames.end(); ++iter ) {
    if( iter->labelName == label ) {
      return true;
    }
  }
  return false;
}

//__________________________________
// Allow the component to set the output interval
void
DataArchiver::updateOutputInterval( double newinv )
{
  if (d_outputInterval != newinv)
  {
    d_outputInterval = newinv;
    d_outputTimestepInterval = 0;
    d_nextOutputTime = 0.0;
  }
}

//__________________________________
// Allow the component to set the output timestep interval
void
DataArchiver::updateOutputTimestepInterval( int newinv )
{
  if (d_outputTimestepInterval != newinv)
  {
    d_outputTimestepInterval = newinv;
    d_outputInterval = 0;
    d_nextOutputTime = 0.0;
  }
}

//__________________________________
// Allow the component to set the checkpoint interval
void
DataArchiver::updateCheckpointInterval( double newinv )
{
  if (d_checkpointInterval != newinv)
  {
    d_checkpointInterval = newinv;
    d_checkpointTimestepInterval = 0;
    d_nextCheckpointTime = 0.0;

    // If needed create checkpoints/index.xml
    if( !d_checkpointsDir.exists() )
    {
      if( d_myworld->myrank() == 0) {
        d_checkpointsDir = d_dir.createSubdir("checkpoints");
        createIndexXML(d_checkpointsDir);
      }
    }

    // Sync up before every rank can use the checkpoints dir
    Uintah::MPI::Barrier(d_myworld->getComm());
  }
}

//__________________________________
// Allow the component to set the checkpoint timestep interval
void
DataArchiver::updateCheckpointTimestepInterval( int newinv )
{
  if (d_checkpointTimestepInterval != newinv)
  {
    d_checkpointTimestepInterval = newinv;
    d_checkpointInterval = 0;
    d_nextCheckpointTime = 0.0;

    // If needed create checkpoints/index.xml
    if( !d_checkpointsDir.exists())
    {
      if( d_myworld->myrank() == 0) {
        d_checkpointsDir = d_dir.createSubdir("checkpoints");
        createIndexXML(d_checkpointsDir);
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
  int dir_timestep = getTimestepTopLevel();  // could be modified by reduceUda
  
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
// If your using reduceUda then use use a mapping that's defined in
// reduceUdaSetup()
int
DataArchiver::getTimestepTopLevel()
{
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  
  if ( d_usingReduceUda ) {
    return d_restartTimestepIndicies[ timestep ];
  }
  else {
    return timestep;
  }
}

//______________________________________________________________________
// Called by In-situ VisIt to dump a time step's data.
void
DataArchiver::outputTimestep( double time,
                              double delt,
                              const GridP& grid,
                              SchedulerP& sched )
{
  int proc = d_myworld->myrank();

  LoadBalancerPort * lb =
    dynamic_cast< LoadBalancerPort * >( getPort( "load balancer" ) );

  DataWarehouse* newDW = sched->getLastDW();
  
  // Save the var so to return to the normal output schedule.
  int nextOutputTimestep = d_nextOutputTimestep;
  int outputTimestepInterval = d_outputTimestepInterval;

  d_nextOutputTimestep = d_sharedState->getCurrentTopLevelTimeStep();;
  d_outputTimestepInterval = 1;

  // Set up the inital bits including the flag d_isOutputTimestep
  // which triggers most actions.
  beginOutputTimestep(time, delt, grid);

  // Updaate the main xml file and write the xml file for this
  // timestep.
  writeto_xml_files(delt, grid);

  // For each level get the patches associated with this processor and
  // save the requested output variables.
  for( int i = 0; i < grid->numLevels(); ++i ) {
    const LevelP& level = grid->getLevel( i );

    const PatchSet* patches = lb->getOutputPerProcessorPatchSet(level);

    outputVariables( nullptr, patches->getSubset(proc), nullptr, nullptr, newDW, OUTPUT );
    outputVariables( nullptr, patches->getSubset(proc), nullptr, nullptr, newDW, CHECKPOINT );
  }

  // Restore the timestep vars so to return to the normal output
  // schedule.
  d_nextOutputTimestep = nextOutputTimestep;
  d_outputTimestepInterval = outputTimestepInterval;

  d_isOutputTimestep = false;
  d_isCheckpointTimestep = false;
}

//______________________________________________________________________
// Called by In-situ VisIt to dump a checkpoint.
//
void
DataArchiver::checkpointTimestep( double time,
                                  double delt,
                                  const GridP& grid,
                                  SchedulerP& sched )
{
  int proc = d_myworld->myrank();

  LoadBalancerPort * lb =
    dynamic_cast< LoadBalancerPort * >( getPort( "load balancer" ) );

  DataWarehouse* newDW = sched->getLastDW();

  // Save the vars so to return to the normal output schedule.
  int nextCheckpointTimestep = d_nextCheckpointTimestep;
  int checkpointTimestepInterval = d_checkpointTimestepInterval;

  d_nextCheckpointTimestep = d_sharedState->getCurrentTopLevelTimeStep();
  d_checkpointTimestepInterval = 1;

  // If needed create checkpoints/index.xml
  if( !d_checkpointsDir.exists()) {
    if( proc == 0) {
      d_checkpointsDir = d_dir.createSubdir("checkpoints");
      createIndexXML( d_checkpointsDir );
    }
  }

  // Sync up before every rank can use the checkpoints dir
  Uintah::MPI::Barrier( d_myworld->getComm() );

  // Set up the inital bits including the flag d_isCheckpointTimestep
  // which triggers most actions.
  beginOutputTimestep( time, delt, grid );

  // Updaate the main xml file and write the xml file for this
  // timestep.
  writeto_xml_files( delt, grid );

  // For each level get the patches associated with this processor and
  // save the requested output variables.
  for( int i = 0; i < grid->numLevels(); ++i ) {
    const LevelP& level = grid->getLevel( i );

    const PatchSet* patches = lb->getOutputPerProcessorPatchSet(level);

    outputVariables( nullptr, patches->getSubset(proc),
		     nullptr, nullptr, newDW, CHECKPOINT );
  }

  // Restore the vars so to return to the normal output schedule.
  d_nextCheckpointTimestep = nextCheckpointTimestep;
  d_checkpointTimestepInterval = checkpointTimestepInterval;

  d_isOutputTimestep = false;
  d_isCheckpointTimestep = false;
}

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
    string svn_diff_out = d_dir.getName() + "/svn_diff.txt";
    string svn_diff_on = string( sci_getenv("SCIRUN_OBJDIR") ) + "/.do_svn_diff";
    if( !validFile( svn_diff_on ) ) {
      cout << "\n"
	   << "WARNING: Adding 'svn diff' file to UDA, "
	   << "but AUTO DIFF TEXT CREATION is OFF!\n"
	   << "         svn_diff.txt may be out of date!  "
	   << "Saving as 'possible_svn_diff.txt'.\n"
	   << "\n";
      
      svn_diff_out = d_dir.getName() + "/possible_svn_diff.txt";
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
  if( d_myworld->myrank() == 0 ) {

    d_writeMeta = true;
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
    d_writeMeta = false; // Only rank 0 will emit meta data...

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

  if( d_myworld->myrank() != 0 ) { // Make sure everyone else can see the temp file...

    struct stat st;
    int s = stat( fs_test_file_name.c_str(), &st );

    if( ( s != 0 ) || !S_ISREG( st.st_mode ) ) {
      cerr << "Stat'ing of file: " << fs_test_file_name
	   << " failed with errno = " << errno << "\n";
      throw ErrnoException( "stat", errno, __FILE__, __LINE__ );
    }
  }

  Uintah::MPI::Barrier(d_myworld->getComm()); // Wait until everyone has check for the file before proceeding.

  if( d_writeMeta ) {

    int s = unlink( fs_test_file_name.c_str() ); // Remove the tmp file...
    if(s != 0) {
      cerr << "Cannot unlink file: " << fs_test_file_name << '\n';
      throw ErrnoException("unlink", errno, __FILE__, __LINE__);
    }

    makeVersionedDir();
    // Send UDA name to all other ranks.
    string udadirname = d_dir.getName();
    
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

    d_dir = Dir( inbuf );
    delete[] inbuf;
  }

  if( d_myworld->myrank() == 0 ) {
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
  if( d_myworld->myrank() == 0 ) {
    // Create a unique string, using hostname+pid
    char* base = strdup(d_filebase.c_str());
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
  myname << basename << "-" << d_myworld->myrank() << ".tmp";
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
  d_writeMeta=true;
  int i;
  for(i=0;i<d_myworld->myrank();i++) {
    ostringstream name;
    name << basename << "-" << i << ".tmp";
    struct stat st;
    int s=stat(name.str().c_str(), &st);
    if(s == 0 && S_ISREG(st.st_mode)) {
      // File exists, we do NOT need to emit metadata
      d_writeMeta=false;
      break;
    }
    else if(errno != ENOENT) {
      cerr << "Cannot stat file: " << name.str() << ", errno=" << errno << '\n';
      throw ErrnoException("stat", errno, __FILE__, __LINE__);
    }
  }

  Uintah::MPI::Barrier( d_myworld->getComm() );

  if( d_writeMeta ) {
    makeVersionedDir();
    string fname = myname.str();
    FILE* tmpout = fopen(fname.c_str(), "w");
    if(!tmpout) {
      throw ErrnoException("fopen", errno, __FILE__, __LINE__);
    }
    string dirname = d_dir.getName();
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

if(!d_writeMeta) {
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
    d_dir = Dir( dirname );
  }

  int count = d_writeMeta ? 1 : 0;
  int nunique;
  // This is an AllReduce, not a reduce.  This is necessary to
  // ensure that all processors wait before they remove the tmp files
  Uintah::MPI::Allreduce(&count, &nunique, 1, MPI_INT, MPI_SUM,
                d_myworld->getComm());
  if( d_myworld->myrank() == 0 ) {
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
