/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#if HAVE_PIDX
#include <CCA/Ports/PIDXOutputContext.h>
#endif
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>

#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/Endian.h>
#include <Core/Util/Environment.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/FileUtils.h>
#include <Core/Util/StringUtil.h>

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

#include <time.h>
#include <sys/param.h>
#include <strings.h>
#include <unistd.h>

//TODO - BJW - if multilevel reduction doesn't work, fix all
//       getMaterialSet(0)

#define PADSIZE    1024L
#define ALL_LEVELS   99

#define OUTPUT               0
#define CHECKPOINT           1
#define CHECKPOINT_REDUCTION 2

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("DataArchiver", false);
static int PIDX_ts = 0;
bool DataArchiver::d_wereSavesAndCheckpointsInitialized = false;

DataArchiver::DataArchiver(const ProcessorGroup* myworld, int udaSuffix)
  : UintahParallelComponent(myworld),
    d_udaSuffix(udaSuffix),
    d_outputLock("DataArchiver output lock")
{
  d_isOutputTimestep      = false;
  d_isCheckpointTimestep  = false;
  d_saveParticleVariables = false;
  d_saveP_x               = false;
  d_particlePositionName  = "p.x";
  d_usingReduceUda        = false;
  //d_currentTime=-1;
  //d_currentTimestep=-1;

  d_XMLIndexDoc = NULL;
  d_CheckpointXMLIndexDoc = NULL;

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
                                  SimulationState * state )
{
   dbg << "Doing ProblemSetup \t\t\t\tDataArchiver"<< endl;
   
   d_sharedState = state;
   d_upsFile = params;
  
   if (params->findBlock("ParticlePosition")) {
     params->findBlock("ParticlePosition")->getAttribute("label",d_particlePositionName);
   }
  
   ProblemSpecP p = params->findBlock("DataArchiver");

   d_outputDoubleAsFloat = p->findBlock("outputDoubleAsFloat") != 0;

   // set to false if restartSetup is called - we can't do it there
   // as the first timestep doesn't have any tasks
   d_outputInitTimestep = p->findBlock("outputInitTimestep") != 0;
   
   // problemSetup is called again from the Switcher to reset vars (and frequency) it wants to save
   //   DO NOT get it again.  Currently the directory won't change mid-run, so calling problemSetup
   //   will not change the directory.  What happens then, is even if a switched component wants a 
   //   different uda name, it will not get one until sus restarts (i.e., when you switch, component
   //   2's data dumps will be in whichever uda started sus.), which is not optimal.  So we disable
   //   this feature until we can make the DataArchiver make a new directory mid-run.
   if (d_filebase == "")
     p->require("filebase", d_filebase);

   // get output timestep or time interval info
   d_outputInterval = 0;
   if (!p->get("outputTimestepInterval", d_outputTimestepInterval))
     d_outputTimestepInterval = 0;
     
   if ( !p->get("outputInterval", d_outputInterval) && d_outputTimestepInterval == 0 )
     d_outputInterval = 0.0; // default

   if ( d_outputInterval != 0.0 && d_outputTimestepInterval != 0 )
     throw ProblemSetupException("Use <outputInterval> or <outputTimestepInterval>, not both",
                                 __FILE__, __LINE__);
   
   // set default compression mode - can be "tryall", "gzip", "rle", "rle, gzip", "gzip, rle", or "none"
   string defaultCompressionMode = "";
   if (p->get("compression", defaultCompressionMode)) {
     VarLabel::setDefaultCompressionMode(defaultCompressionMode);
   }
   
   // get the variables to save
   d_saveLabelNames.clear(); // we can problemSetup multiple times on a component Switch, clear the old ones.
   map<string, string> attributes;
   SaveNameItem saveItem;
   ProblemSpecP save = p->findBlock("save");

   if( save == 0 ) {
     proc0cout << "\nWARNING: No data will be saved as none was specified to be saved in the .ups file!\n\n";
   }

   while( save != 0 ) {
      attributes.clear();
      save->getAttributes(attributes);
      saveItem.labelName = attributes["label"];
      saveItem.compressionMode = attributes["compression"];
      try {
        saveItem.matls = ConsecutiveRangeSet(attributes["material"]);
      }
      catch (ConsecutiveRangeSetException) {
        throw ProblemSetupException("'" + attributes["material"] + "'" +
               " cannot be parsed as a set of material" +
               " indices for saving '" + saveItem.labelName + "'",
                                    __FILE__, __LINE__);
      }

      if (saveItem.matls.size() == 0)
        // if materials aren't specified, all valid materials will be saved
        saveItem.matls = ConsecutiveRangeSet::all;
      
      try {
        // level values are normal for exact level values (i.e.,
        //   0, 2-4 will save those levels.
        //   Leave blank for all levels
        //   Also handle negative values to be relative to the finest levels
        //   I.e., -3--1 would be the top three levels.
        saveItem.levels = ConsecutiveRangeSet(attributes["levels"]);
      }
      catch (ConsecutiveRangeSetException) {
        throw ProblemSetupException("'" + attributes["levels"] + "'" +
               " cannot be parsed as a set of levels" +
               " for saving '" + saveItem.labelName + "'",
                                    __FILE__, __LINE__);
      }

      if (saveItem.levels.size() == 0)
        // if materials aren't specified, all valid materials will be saved
        saveItem.levels = ConsecutiveRangeSet(ALL_LEVELS, ALL_LEVELS);
      
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
      
      d_saveLabelNames.push_back(saveItem);
      
      save = save->findNextBlock("save");
   }
   if(d_saveP_x == false && d_saveParticleVariables == true) {
     throw ProblemSetupException(" You must save " + d_particlePositionName + " when saving other particle variables",
                                 __FILE__, __LINE__);
   }     
   
   // get checkpoint information
   d_checkpointInterval         = 0.0;
   d_checkpointTimestepInterval = 0;
   d_checkpointWalltimeStart    = 0;
   d_checkpointWalltimeInterval = 0;
   d_checkpointCycle            = 2; /* 2 is the smallest number that is safe
                                     (always keeping an older copy for backup) */
                          
   ProblemSpecP checkpoint = p->findBlock("checkpoint");
   if( checkpoint != 0 ) {

     string interval, timestepInterval, walltimeStart, walltimeInterval, walltimeStartHours, walltimeIntervalHours, cycle;

     attributes.clear();
     checkpoint->getAttributes( attributes );
      
     interval              = attributes[ "interval" ];
     timestepInterval      = attributes[ "timestepInterval" ];
     walltimeStart         = attributes[ "walltimeStart" ];
     walltimeInterval      = attributes[ "walltimeInterval" ];
     walltimeStartHours    = attributes[ "walltimeStartHours" ];
     walltimeIntervalHours = attributes[ "walltimeIntervalHours" ];
     cycle                 = attributes[ "cycle" ];

     if( interval != "" ) {
       d_checkpointInterval = atof( interval.c_str() );
     }
     if( timestepInterval != "" ) {
       d_checkpointTimestepInterval = atoi( timestepInterval.c_str() );
     }
     if( walltimeStart != "" ) {
       d_checkpointWalltimeStart = atof( walltimeStart.c_str() );
     }      
     if( walltimeInterval != "" ) {
       d_checkpointWalltimeInterval = atof( walltimeInterval.c_str() );
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

     // Verify that an interval was specified:
     if( interval == "" && timestepInterval == "" && walltimeInterval == "" && walltimeIntervalHours == "" ) {
       throw ProblemSetupException( "ERROR: \n  <checkpoint> must specify either interval, timestepInterval, walltimeInterval",
                                    __FILE__, __LINE__ );
     }
   }

   // Can't use both checkpointInterval and checkpointTimestepInterval.
   if (d_checkpointInterval != 0.0 && d_checkpointTimestepInterval != 0) {
     throw ProblemSetupException("Use <checkpoint interval=...> or <checkpoint timestepInterval=...>, not both",
                                 __FILE__, __LINE__);
   }
   // Can't have a walltimeStart without a walltimeInterval.
   if (d_checkpointWalltimeStart != 0 && d_checkpointWalltimeInterval == 0) {
     throw ProblemSetupException("<checkpoint walltimeStart must have a corresponding walltimeInterval",
                                 __FILE__, __LINE__);
   }
   // Set walltimeStart to walltimeInterval if not specified.
   if (d_checkpointWalltimeInterval != 0 && d_checkpointWalltimeStart == 0) {
     d_checkpointWalltimeStart = d_checkpointWalltimeInterval;
   }
   //d_currentTimestep = 0;
   
   //int startTimestep;
   //if (p->get("startTimestep", startTimestep)) {
   //  d_currentTimestep = startTimestep - 1;
   //}
  
   d_lastTimestepLocation   = "invalid";
   d_isOutputTimestep       = false;

   // Set up the next output and checkpoint time.
   d_nextOutputTime         = 0.0;
   d_nextOutputTimestep     = d_outputInitTimestep?0:1;
   d_nextCheckpointTime     = d_checkpointInterval; 
   d_nextCheckpointTimestep = d_checkpointTimestepInterval+1;

   proc0cout << "Next checkpoint time is " << d_checkpointInterval << "\n";

   if (d_checkpointWalltimeInterval > 0) {
     d_nextCheckpointWalltime = d_checkpointWalltimeStart + (int) Time::currentSeconds();
     if( Parallel::usingMPI() ) {
       // Make sure we are all writing at same time.  When node clocks disagree,
       // make decision based on processor zero time.
       MPI_Bcast(&d_nextCheckpointWalltime, 1, MPI_INT, 0, d_myworld->getComm());
     }
   }
   else { 
     d_nextCheckpointWalltime = 0;
   }
}
//______________________________________________________________________
//
void
DataArchiver::initializeOutput(const ProblemSpecP& params) 
{
   if( d_outputInterval == 0.0 && 
       d_outputTimestepInterval == 0 && 
       d_checkpointInterval == 0.0 && 
       d_checkpointTimestepInterval == 0 && 
       d_checkpointWalltimeInterval == 0) {
     return;
   }

   if( Parallel::usingMPI() ) {
     // See how many shared filesystems that we have
     double start=Time::currentSeconds();
     string basename;
     if(d_myworld->myrank() == 0) {
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
       if (*base)
         free(base);

       string test_string = ts.str();
       const char* outbuf = test_string.c_str();
       int outlen = (int)strlen(outbuf);

       MPI_Bcast(&outlen, 1, MPI_INT, 0, d_myworld->getComm());
       MPI_Bcast(const_cast<char*>(outbuf), outlen, MPI_CHAR, 0,
                 d_myworld->getComm());
       basename = test_string;
     } else {
       int inlen;
       MPI_Bcast(&inlen, 1, MPI_INT, 0, d_myworld->getComm());
       char* inbuf = scinew char[inlen+1];
       MPI_Bcast(inbuf, inlen, MPI_CHAR, 0, d_myworld->getComm());
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
     if(!tmpout)
       throw ErrnoException("fopen failed for " + fname, errno, __FILE__, __LINE__);
     fprintf(tmpout, "\n");
     if(fflush(tmpout) != 0)
       throw ErrnoException("fflush", errno, __FILE__, __LINE__);
     if(fsync(fileno(tmpout)) != 0)
       throw ErrnoException("fsync", errno, __FILE__, __LINE__);
     if(fclose(tmpout) != 0)
       throw ErrnoException("fclose", errno, __FILE__, __LINE__);
     MPI_Barrier(d_myworld->getComm());
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
       } else if(errno != ENOENT) {
         cerr << "Cannot stat file: " << name.str() << ", errno=" << errno << '\n';
         throw ErrnoException("stat", errno, __FILE__, __LINE__);
       }
     }
     MPI_Barrier(d_myworld->getComm());
     if(d_writeMeta) {
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
     MPI_Barrier(d_myworld->getComm());
     if(!d_writeMeta) {
       ostringstream name;
       name << basename << "-" << i << ".tmp";
       ifstream in(name.str().c_str()); 
       if (!in) {
         throw InternalError("DataArchiver::initializeOutput(): The file \"" + \
                             name.str() + "\" not found on second pass for filesystem discovery!",
                             __FILE__, __LINE__);
       }
       string dirname;
       in >> dirname;
       d_dir=Dir(dirname);
     }
     int count=d_writeMeta?1:0;
     int nunique;
     // This is an AllReduce, not a reduce.  This is necessary to
     // ensure that all processors wait before they remove the tmp files
     MPI_Allreduce(&count, &nunique, 1, MPI_INT, MPI_SUM,
                   d_myworld->getComm());
     if(d_myworld->myrank() == 0) {
       double dt=Time::currentSeconds()-start;
       cerr << "Discovered " << nunique << " unique filesystems in " << dt << " seconds\n";
     }
     // Remove the tmp files...
     int s = unlink(myname.str().c_str());
     if(s != 0) {
       cerr << "Cannot unlink file: " << myname.str() << '\n';
       throw ErrnoException("unlink", errno, __FILE__, __LINE__);
     }
   } else {
      makeVersionedDir();
      d_writeMeta = true;
   }

   if (d_writeMeta) {

     string svn_diff_file = string( sci_getenv("SCIRUN_OBJDIR") ) + "/svn_diff.txt";
     if( !validFile( svn_diff_file ) ) {
       cout << "\n";
       cout << "WARNING: 'svn diff' file '" << svn_diff_file << "' does not appear to exist!\n";
       cout << "\n";
     } 
     else {
       string svn_diff_out = d_dir.getName() + "/svn_diff.txt";
       string svn_diff_on = string( sci_getenv("SCIRUN_OBJDIR") ) + "/.do_svn_diff";
       if( !validFile( svn_diff_on ) ) {
         cout << "\n";
         cout << "WARNING: Adding 'svn diff' file to UDA, but AUTO DIFF TEXT CREATION is OFF!\n";
         cout << "         svn_diff.txt may be out of date!  Saving as 'possible_svn_diff.txt'.\n";
         cout << "\n";
         svn_diff_out = d_dir.getName() + "/possible_svn_diff.txt";
       }
       copyFile( svn_diff_file, svn_diff_out );
     }

      // create index.xml 
      string inputname = d_dir.getName()+"/input.xml";
      params->output(inputname.c_str());

      dynamic_cast<SimulationInterface*>(getPort("sim"))->outputPS(d_dir);

      /////////////////////////////////////////////////////////
      // Save the original .ups file in the UDA...
      //     FIXME: might want to avoid using 'system' copy which the below uses...
      //     If so, we will need to write our own (simple) file reader and writer
      //     routine.

      cout << "Saving original .ups file in UDA...\n";
      Dir ups_location( pathname( params->getFile() ) );
      ups_location.copy( basename( params->getFile() ), d_dir );

      //
      /////////////////////////////////////////////////////////

      createIndexXML(d_dir);
   
      // create checkpoints/index.xml (if we are saving checkpoints)
      if (d_checkpointInterval != 0.0 || d_checkpointTimestepInterval != 0 ||
          d_checkpointWalltimeInterval != 0) {
         d_checkpointsDir = d_dir.createSubdir("checkpoints");
         createIndexXML(d_checkpointsDir);
      }
   }
   else {
      d_checkpointsDir = d_dir.getSubdir("checkpoints");
   }

   //sync up before every rank can use the base dir
   if (Parallel::usingMPI()) { 
       MPI_Barrier(d_myworld->getComm());
   }
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
      // the restart_merger doesn't need checkpoints, and calls this with time=0.
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
          if( Parallel::usingMPI() ) {
            printf( "WARNING: Filesystem check failed on processor %d\n", Parallel::getMPIRank() );
          } else {
            printf( "WARNING: The filesystem appears to be flaky...\n" );
          }
        }
        // Verify that "system works"
        int code = system( "echo how_are_you" );
        if (code != 0) {
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
   
  // set time and timestep variables appropriately
  //d_currentTimestep = timestep;
   
  if( d_outputInterval > 0 ) {
    d_nextOutputTime = d_outputInterval * ceil(time / d_outputInterval);
  }
  else if( d_outputTimestepInterval > 0 ) {
    d_nextOutputTimestep = (timestep/d_outputTimestepInterval) * d_outputTimestepInterval + 1;
    while( d_nextOutputTimestep <= timestep ) {
      d_nextOutputTimestep+=d_outputTimestepInterval;
    }
  }
   
  if( d_checkpointInterval > 0 ) {
    d_nextCheckpointTime = d_checkpointInterval * ceil(time / d_checkpointInterval);
  }
  else if( d_checkpointTimestepInterval > 0 ) {
    d_nextCheckpointTimestep = (timestep/d_checkpointTimestepInterval)*d_checkpointTimestepInterval+1;
    while( d_nextCheckpointTimestep <= timestep ) {
      d_nextCheckpointTimestep += d_checkpointTimestepInterval;
    }
  }
  if( d_checkpointWalltimeInterval > 0 ) {
    d_nextCheckpointWalltime = d_checkpointWalltimeInterval + (int)Time::currentSeconds();
    if(Parallel::usingMPI()) {
      MPI_Bcast(&d_nextCheckpointWalltime, 1, MPI_INT, 0, d_myworld->getComm());
    }
  }
}

//______________________________________________________________________
// This is called after problemSetup. It will copy the dat & checkpoint files to the new directory.
// This also removes the global (dat) variables from the saveLabels variables
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
    proc0cout << "*** Copied dat files to:   " << d_dir.getName() << endl;
    
    // copy checkpoints
    Dir checkpointsFromDir = fromDir.getSubdir("checkpoints");
    Dir checkpointsToDir   = d_dir.getSubdir("checkpoints");
    string me = checkpointsFromDir.getName();
    if( validDir(me) ) {
      checkpointsToDir.remove( "index.xml", false);  // this file is created upstream when it shouldn't have
      checkpointsFromDir.copy( d_dir );
      proc0cout << "\n*** Copied checkpoints to: " << d_checkpointsDir.getName() << endl;
      proc0cout << "    Only using 1 processor to copy so this will be slow for large checkpoint directories\n" << endl;
    }

    // copy input.xml.orig if it exists
    string there = d_dir.getName();
    string here  = fromDir.getName() + "/input.xml.orig";
    if ( validFile(here) ) {
      fromDir.copy("input.xml.orig", d_dir);     // use OS independent copy functions, needed by mira
      proc0cout << "*** Copied input.xml.orig to: " << there << endl;
    }
    
    // copy the original ups file if it exists
    vector<string> ups;
    fromDir.getFilenamesBySuffix( "ups", ups );
    
    if ( ups.size() != 0 ) {
      fromDir.copy(ups[0], d_dir);              // use OS independent copy functions, needed by mira
      proc0cout << "*** Copied ups file ("<< ups[0]<< ") to: " << there << endl;
    }
    proc0cout << "\n"<<endl;
  }

  //__________________________________
  //
  // removed the global (dat) variables from the saveLabels
  string iname = fromDir.getName()+"/index.xml";
  ProblemSpecP indexDoc = loadDocument(iname);

  ProblemSpecP globals = indexDoc->findBlock("globals");
  if (globals != 0) {

    ProblemSpecP variable = globals->findBlock("variable");
    while (variable != 0) {
      string varname;

      if ( !variable->getAttribute("name", varname) ) {
        throw InternalError("global variable name attribute not found", __FILE__, __LINE__);
      }

      list<SaveNameItem>::iterator it = d_saveLabelNames.begin();
      while ( it != d_saveLabelNames.end() ) {
        if ( (*it).labelName == varname ) {
          it = d_saveLabelNames.erase(it);
        }
        else {
          it++;
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
  
  while( ts != 0 ) {
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
}

//______________________________________________________________________
//
void
DataArchiver::copySection(Dir& fromDir, Dir& toDir, string filename, string section)
{
  // copy chunk labeled section between index.xml files
  string iname = fromDir.getName() + "/" +filename;
  ProblemSpecP indexDoc = loadDocument(iname);

  iname = toDir.getName() + "/" + filename;
  ProblemSpecP myIndexDoc = loadDocument(iname);
  
  ProblemSpecP sectionNode = indexDoc->findBlock(section);
  if (sectionNode != 0) {
    ProblemSpecP newNode = myIndexDoc->importNode(sectionNode, true);
    
    // replace whatever was in the section previously
    ProblemSpecP mySectionNode = myIndexDoc->findBlock(section);
    if (mySectionNode != 0) {
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
   if( restarts == 0 ) {
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
   if (oldTimesteps != 0)
     ts = oldTimesteps->findBlock("timestep");

   // while we're at it, add restart information to index.xml
   if (maxTimestep >= 0)
     addRestartStamp(indexDoc, fromDir, maxTimestep);

   // create timesteps element if necessary
   ProblemSpecP timesteps = indexDoc->findBlock("timesteps");
   if (timesteps == 0) {
      timesteps = indexDoc->appendChild("timesteps");
   }
   
   // copy each timestep 
   int timestep;
   while (ts != 0) {
      ts->get(timestep);
      if (timestep >= startTimestep &&
          (timestep <= maxTimestep || maxTimestep < 0)) {
         // copy the timestep directory over
         map<string,string> attributes;
         ts->getAttributes(attributes);

         string hrefNode = attributes["href"];
         if (hrefNode == "")
            throw InternalError("timestep href attribute not found", __FILE__, __LINE__);

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
         ProblemSpecP newTS = timesteps->appendElement("timestep", timestep_str.str().c_str());

         for (map<string,string>::iterator iter = attributes.begin();
              iter != attributes.end(); iter++) {
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
   if (globals != 0) {
      ProblemSpecP variable = globals->findBlock("variable");
      // copy data file associated with each variable
      while (variable != 0) {
         map<string,string> attributes;
         variable->getAttributes(attributes);

         string hrefNode = attributes["href"];

         if (hrefNode == "")
            throw InternalError("global variable href attribute not found", __FILE__, __LINE__);
         const char* href = hrefNode.c_str();

         ifstream datFile((fromDir.getName()+"/"+href).c_str());
         if (!datFile) {
           throw InternalError("DataArchiver::copyDatFiles(): The file \"" + \
                               (fromDir.getName()+"/"+href) + \
                               "\" could not be opened for reading!", __FILE__, __LINE__);
         }
         ofstream copyDatFile((toDir.getName()+"/"+href).c_str(), ios::app);
         if (!copyDatFile) {
           throw InternalError("DataArchiver::copyDatFiles(): The file \"" + \
                               (toDir.getName()+"/"+href) + \
                               "\" could not be opened for writing!", __FILE__, __LINE__);
         }

         // copy up to maxTimestep lines of the old dat file to the copy
         int timestep = startTimestep;
         while (datFile.getline(buffer, 1000) &&
                (timestep < maxTimestep || maxTimestep < 0)) {
            copyDatFile << buffer << endl;
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
 
   ProblemSpecP metaElem = rootElem->appendChild("Meta");

   // Some systems dont supply a logname
   const char * logname = getenv("LOGNAME");
   if(logname) metaElem->appendElement("username", logname);

   time_t t = time(NULL) ;
   
   // Chop the newline character off the time string so that the Date
   // field will appear properly in the XML
   string time_string(ctime(&t));
   string::iterator end = time_string.end();
   --end;
   time_string.erase(end);
   metaElem->appendElement("date", time_string.c_str());
   metaElem->appendElement("endianness", endianness().c_str());
   metaElem->appendElement("nBits", (int)sizeof(unsigned long) * 8 );
   
   string iname = dir.getName()+"/index.xml";
   rootElem->output(iname.c_str());
   //rootElem->releaseDocument();
}


//______________________________________________________________________
//
void
DataArchiver::finalizeTimestep(double time, 
                               double delt,
                               const GridP& grid, 
                               SchedulerP& sched,
                               bool recompile /*=false*/)
{
  //this function should get called exactly once per timestep
  
  //  static bool wereSavesAndCheckpointsInitialized = false;
  dbg << "  finalizeTimestep, delt= " << delt << endl;
  d_tempElapsedTime = time+delt;
  
  beginOutputTimestep(time, delt, grid);

  //__________________________________
  // some changes here - we need to redo this if we add a material, or if we schedule output
  // on the initialization timestep (because there will be new computes on subsequent timestep)
  // or if there is a component switch or a new level in the grid
  // - BJW
  if (((delt != 0 || d_outputInitTimestep) && !d_wereSavesAndCheckpointsInitialized) || 
        d_sharedState->d_switchState || grid->numLevels() != d_numLevelsInOutput) {
      /* skip the initialization timestep (normally, anyway) for this
         because it needs all computes to be set
         to find the save labels */
    
    if (d_outputInterval != 0.0 || d_outputTimestepInterval != 0) {
      initSaveLabels(sched, delt == 0);
     
      if (!d_wereSavesAndCheckpointsInitialized && delt != 0) {
        indexAddGlobals(); /* add saved global (reduction) variables to index.xml */
      }
    }
    
    // This assumes that the TaskGraph doesn't change after the second
    // timestep and will need to change if the TaskGraph becomes dynamic. 
    //   We also need to do this again if this is the initial timestep
    if (delt != 0) {
      d_wereSavesAndCheckpointsInitialized = true;
    
      // can't do checkpoints on init timestep....
      if (d_checkpointInterval != 0.0 || d_checkpointTimestepInterval != 0 || 
          d_checkpointWalltimeInterval != 0) {

        initCheckpoints(sched);
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
DataArchiver::sched_allOutputTasks(double delt,
                                   const GridP& grid, 
                                   SchedulerP& sched,
                                   bool recompile /*=false*/)
{
  dbg << "  sched_allOutputTasks \n";
  
  // we don't want to schedule more tasks unless we're recompiling
  if ( !recompile ) {
    return;
  }

  //__________________________________
  //  Reduction Variables
  // Schedule task to dump out reduction variables at every timestep
  
  if ( (d_outputInterval != 0.0 || d_outputTimestepInterval != 0) &&
       (delt != 0 || d_outputInitTimestep)) {
    
    Task* t = scinew Task("DataArchiver::outputReductionVars",this, 
                          &DataArchiver::outputReductionVars);
    
    for(int i=0;i<(int)d_saveReductionLabels.size();i++) {
      SaveItem& saveItem = d_saveReductionLabels[i];
      const VarLabel* var = saveItem.label;
      map<int, MaterialSetP>::iterator liter;
      
      for (liter = saveItem.matlSet.begin(); liter != saveItem.matlSet.end(); liter++) {
        const MaterialSubset* matls = saveItem.getMaterialSet(liter->first)->getUnion();
        t->requires(Task::NewDW, var, matls, true);
        break; // this might break things later, but we'll leave it for now
      }
    }
    
    sched->addTask(t, 0, 0);
    
    dbg << "  scheduled output tasks (reduction variables)" << endl;
    if (delt != 0 || d_outputInitTimestep) {
      scheduleOutputTimestep(d_saveLabels, grid, sched, false);
    }
  }
  
  //__________________________________
  //  Schedule Checkpoint (reduction variables)
  if (delt != 0 && d_checkpointCycle>0 &&
      (d_checkpointInterval>0 || d_checkpointTimestepInterval>0 || d_checkpointWalltimeInterval>0 ) ) {
    // output checkpoint timestep
    Task* t = scinew Task("DataArchiver::outputVariables (CheckpointReduction)",this, 
                          &DataArchiver::outputVariables, CHECKPOINT_REDUCTION);
    
    for(int i=0;i<(int)d_checkpointReductionLabels.size();i++) {
      SaveItem& saveItem = d_checkpointReductionLabels[i];
      const VarLabel* var = saveItem.label;
      map<int, MaterialSetP>::iterator liter;
      
      for (liter = saveItem.matlSet.begin(); liter != saveItem.matlSet.end(); liter++) {
        const MaterialSubset* matls = saveItem.getMaterialSet(liter->first)->getUnion();
        t->requires(Task::NewDW, var, matls, true);
        break;
      }
    }
    sched->addTask(t, 0, 0);
    
    dbg << "  scheduled output tasks (checkpoint variables)" << endl;
    
    scheduleOutputTimestep(d_checkpointLabels,  grid, sched, true);
  }
}


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

  // do *not* update d_nextOutputTime or others here.  We need the original
  // values to compare if there is a timestep restart.  See 
  // reEvaluateOutputTimestep
  if (d_outputInterval != 0.0 && (delt != 0 || d_outputInitTimestep)) {
    if(time+delt >= d_nextOutputTime) {
      // output timestep
      d_isOutputTimestep = true;
      makeTimestepDirs(d_dir, d_saveLabels, grid, &d_lastTimestepLocation);
    }
    else {
      d_isOutputTimestep = false;
    }
  }
  else if (d_outputTimestepInterval != 0 && (delt != 0 || d_outputInitTimestep)) {
    if(timestep >= d_nextOutputTimestep) {
      // output timestep
      d_isOutputTimestep = true;
      makeTimestepDirs(d_dir, d_saveLabels, grid, &d_lastTimestepLocation);
    }
    else {
      d_isOutputTimestep = false;
    }
  }
  
  int currsecs = (int)Time::currentSeconds();
  if(Parallel::usingMPI() && d_checkpointWalltimeInterval != 0) {
     MPI_Bcast(&currsecs, 1, MPI_INT, 0, d_myworld->getComm());
  }
  
  //__________________________________
  // same thing for checkpoints
  if( ( d_checkpointInterval != 0.0 && time+delt >= d_nextCheckpointTime ) ||
      ( d_checkpointTimestepInterval != 0 && timestep >= d_nextCheckpointTimestep ) ||
      ( d_checkpointWalltimeInterval != 0 && currsecs >= d_nextCheckpointWalltime ) ) {

    d_isCheckpointTimestep=true;

    string timestepDir;
    makeTimestepDirs(d_checkpointsDir, d_checkpointLabels, grid, &timestepDir );
    
    string iname = d_checkpointsDir.getName()+"/index.xml";

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
            if( Parallel::usingMPI() ) {
              printf( "WARNING: Filesystem check failed on processor %d\n", Parallel::getMPIRank() );
            } else {
              printf( "WARNING: The filesystem appears to be flaky...\n" );
            }
          }
        }
      }
      d_checkpointTimestepDirs.pop_front();
    }
    //if (d_writeMeta)
    //index->releaseDocument();
  } else {
    d_isCheckpointTimestep=false;
  }
  dbg << "    write CheckPoints (" << d_isCheckpointTimestep << ")  write output timestep (" << d_isOutputTimestep << ")";
  dbg << "    end\n";
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

  // Create the directory for this timestep, if necessary
  // It is not gurantteed that the rank holding d_writeMeta will call 
  // outputTimstep to create dir before another rank begin to output data.
  // A race condition happens when a rank executes output task and
  // the rank holding d_writeMeta is still compiling task graph. 
  // So every rank should try to create dir.
  //if(d_writeMeta) {
    Dir tdir;

    bool done = false;
    int  tries = 0;

    while( !done ) {

      try {
        tries++;

        if( tries > 500 ) {
          cout << "Tries is " << tries << "\n";
          throw InternalError( "DataArchiver::outputTimstep(): Unable to create timestep directory!", __FILE__, __LINE__ );
        }

        tdir = baseDir.createSubdir(tname.str());
        done = true;
      }
      catch( ErrnoException & e ) {
        if( e.getErrno() == EEXIST ) {
          done = true;
          tdir = baseDir.getSubdir( tname.str() );
        }
      }
    }

    if( tries > 1 ) {
      cout << d_myworld->myrank() << " - tries: " << tries << "\n";
    }
    
    // Create the directory for this level, if necessary

    for( int l = 0; l < numLevels; l++ ) {

      tries = 0;
      ostringstream lname;
      lname << "l" << l;
      Dir ldir;
      
      done = false;
      while( !done ) {

        try {
          tries++;

          if( tries > 500 ) {
            cout << "2) Tries is " << tries << "\n";
            throw InternalError( "DataArchiver::outputTimstep(): 2) Unable to create timestep directory!", __FILE__, __LINE__ );
          }
          
          ldir = tdir.createSubdir( lname.str() );
          done = true;
        }
        catch( ErrnoException & e ) {
          if( e.getErrno() == EEXIST ) {
            done = true;
            ldir = tdir.getSubdir( lname.str() );
          }
        }
      }
      if( tries > 1 ) {
        cout << d_myworld->myrank() << ": " << l << " - tries: " << tries << "\n";
      }

    } // end for( int l = 0 )
//}
}


//______________________________________________________________________
//
void
DataArchiver::reEvaluateOutputTimestep(double /*orig_delt*/, double new_delt)
{
  dbg << "  reEvaluateOutputTimestep() begin\n";
  // call this on a timestep restart.  If lowering the delt goes beneath the 
  // threshold, mark it as not an output timestep

  // this is set in finalizeTimestep to time+delt
  d_tempElapsedTime = d_sharedState->getElapsedTime() + new_delt;

  if (d_isOutputTimestep && d_outputInterval != 0.0 ) {
    if (d_tempElapsedTime < d_nextOutputTime)
      d_isOutputTimestep = false;
  }
  if (d_isCheckpointTimestep && d_checkpointInterval != 0.0) {
    if (d_tempElapsedTime < d_nextCheckpointTime) {
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
  dbg << "  reEvaluateOutputTimestep() end\n";
}

//______________________________________________________________________
//
void
DataArchiver::findNext_OutputCheckPoint_Timestep(double delt, const GridP& grid)
{
  dbg << "  findNext_OutputCheckPoint_Timestep() begin\n";
  // double time = d_sharedState->getElapsedTime();
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  // if this was an output/checkpoint timestep,
  // determine when the next one will be.

  // don't do this in beginOutputTimestep because the timestep might restart
  // and we need to have the output happen exactly when we need it.
  if (d_isOutputTimestep) {
    if (d_outputInterval != 0.0) {
      // output timestep
      if(d_tempElapsedTime >= d_nextOutputTime)
        d_nextOutputTime+=floor((d_tempElapsedTime-d_nextOutputTime)/d_outputInterval)*d_outputInterval+d_outputInterval;
    }
    else if (d_outputTimestepInterval != 0) {
      if(timestep>=d_nextOutputTimestep)
      {
        d_nextOutputTimestep+=((timestep-d_nextOutputTimestep)/d_outputTimestepInterval)*d_outputTimestepInterval+d_outputTimestepInterval;
      }
    }
  }

  if (d_isCheckpointTimestep) {
    if (d_checkpointInterval != 0.0) {
      if(d_tempElapsedTime >= d_nextCheckpointTime)
        d_nextCheckpointTime+=floor((d_tempElapsedTime-d_nextCheckpointTime)/d_checkpointInterval)*d_checkpointInterval+d_checkpointInterval;
    }
    else if (d_checkpointTimestepInterval != 0) {
      if(timestep >= d_nextCheckpointTimestep)
        d_nextCheckpointTimestep+=((timestep-d_nextCheckpointTimestep)/d_checkpointTimestepInterval)*d_checkpointTimestepInterval+d_checkpointTimestepInterval;
    }
    if (d_checkpointWalltimeInterval != 0) {
      if(Time::currentSeconds() >= d_nextCheckpointWalltime)
        d_nextCheckpointWalltime+=static_cast<int>(floor((Time::currentSeconds()-d_nextCheckpointWalltime)/d_checkpointWalltimeInterval)*d_checkpointWalltimeInterval+d_checkpointWalltimeInterval);
    }
  }  
  
  dbg << "    next outputTime:     " << d_nextOutputTime << " next outputTimestep: " << d_nextOutputTimestep << "\n";
  dbg << "    next checkpointTime: " << d_nextCheckpointTime << " next checkpoint timestep: " << d_nextCheckpointTimestep << "\n";
  dbg << "  end\n";
  
}


//______________________________________________________________________
//  update the xml files for the 
void
DataArchiver::writeto_xml_files(double delt, const GridP& grid)
{
  dbg << "  writeto_xml_files() begin\n";
  //__________________________________
  //  Writeto XML files
  // to check for output nth proc
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(getPort("load balancer"));
  int dir_timestep = getTimestepTopLevel();     // could be modified by reduceUda
  
  // start dumping files to disk
  vector<Dir*> baseDirs;
  if (d_isOutputTimestep) {
    baseDirs.push_back(&d_dir);
  }    
  if (d_isCheckpointTimestep) {
    baseDirs.push_back(&d_checkpointsDir);
  }

  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << dir_timestep;

  for (int i = 0; i < static_cast<int>(baseDirs.size()); i++) {
    // to save the list of vars. up to 2, since in checkpoints, there are two types of vars
    vector<vector<SaveItem>*> savelist; 
    
    // Reference this timestep in index.xml
    if(d_writeMeta) {
      string iname = baseDirs[i]->getName()+"/index.xml";

      ProblemSpecP indexDoc;
      bool hasGlobals = false;

      if ( baseDirs[i] == &d_dir ) {
        savelist.push_back(&d_saveLabels);
      }
      else if ( baseDirs[i] == &d_checkpointsDir ) {
        hasGlobals = d_checkpointReductionLabels.size() > 0;
        savelist.push_back(&d_checkpointLabels);
        savelist.push_back(&d_checkpointReductionLabels);
      }
      else {
        throw "DataArchiver::writeto_xml_files(): Unknown directory!";
      }
      indexDoc = loadDocument(iname);

      // if this timestep isn't already in index.xml, add it in
      if (indexDoc == 0)
        continue; // output timestep but no variables scheduled to be saved.
      ASSERT(indexDoc != 0);

      // output data pointers
      for (unsigned j = 0; j < savelist.size(); j++) {
        string variableSection = savelist[j] == &d_checkpointReductionLabels ? "globals" : "variables";
        ProblemSpecP vs = indexDoc->findBlock(variableSection);
        if(vs == 0) {
          vs = indexDoc->appendChild(variableSection.c_str());
        }
        for (unsigned k = 0; k < savelist[j]->size(); k++) {
          const VarLabel* var = (*savelist[j])[k].label;
          bool found=false;
          for(ProblemSpecP n = vs->getFirstChild(); n != 0; n=n->getNextSibling()) {
            if(n->getNodeName() == "variable") {
              map<string,string> attributes;
              n->getAttributes(attributes);
              string varname = attributes["name"];
              if(varname == "")
                throw InternalError("varname not found", __FILE__, __LINE__);
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
      
      // Check if it's the first checkpoint timestep by checking if the "timesteps" field is in 
      // checkpoints/index.xml.  If it is then there exists a timestep.xml file already.
      // Use this below to change information in input.xml...
      bool firstCheckpointTimestep = false;
      
      ProblemSpecP ts = indexDoc->findBlock("timesteps");
      if(ts == 0) {
        ts = indexDoc->appendChild("timesteps");
        firstCheckpointTimestep = (&d_checkpointsDir == baseDirs[i]);
      }
      bool found=false;
      for(ProblemSpecP n = ts->getFirstChild(); n != 0; n=n->getNextSibling()) {
        if(n->getNodeName() == "timestep") {
          int readtimestep;
          if(!n->get(readtimestep))
            throw InternalError("Error parsing timestep number", __FILE__, __LINE__);
          if(readtimestep == dir_timestep) {
            found=true;
            break;
          }
        }
      }
      if(!found) {
        // add timestep info
        string timestepindex = tname.str()+"/timestep.xml";      
        
        ostringstream value, timeVal, deltVal;
        value << dir_timestep;
        ProblemSpecP newElem = ts->appendElement("timestep",value.str().c_str());
        newElem->setAttribute("href", timestepindex.c_str());
        timeVal << std::setprecision(17) << d_tempElapsedTime;
        newElem->setAttribute("time", timeVal.str());
        deltVal << std::setprecision(17) << delt;
        newElem->setAttribute("oldDelt", deltVal.str());
      }
      
      indexDoc->output(iname.c_str());
      //indexDoc->releaseDocument();

      // make a timestep.xml file for this timestep 
      // we need to do it here in case there is a timestesp restart
      ProblemSpecP rootElem = ProblemSpec::createDocument("Uintah_timestep");


      // Create a metadata element to store the per-timestep endianness
      ProblemSpecP metaElem = rootElem->appendChild("Meta");
      metaElem->appendElement("endianness", endianness().c_str());
      metaElem->appendElement("nBits", (int)sizeof(unsigned long) * 8 );
      metaElem->appendElement("numProcs", d_myworld->size());
      

      ProblemSpecP timeElem = rootElem->appendChild("Time");
      timeElem->appendElement("timestepNumber", dir_timestep);
      timeElem->appendElement("currentTime", d_tempElapsedTime);
      timeElem->appendElement("oldDelt", delt);
      int numLevels = grid->numLevels();
      
      // in amr, we're not guaranteed that a proc do work on a given level
      //   quick check to see that, so we don't create a node that points to no data
      vector<vector<int> > procOnLevel(numLevels);

      ProblemSpecP gridElem = rootElem->appendChild("Grid");
      gridElem->appendElement("numLevels", numLevels);
      for(int l = 0;l<numLevels;l++) {
        LevelP level = grid->getLevel(l);
        ProblemSpecP levelElem = gridElem->appendChild("Level");

        if (level->getPeriodicBoundaries() != IntVector(0,0,0))
          levelElem->appendElement("periodic", level->getPeriodicBoundaries());
        levelElem->appendElement("numPatches", level->numPatches());
        levelElem->appendElement("totalCells", level->totalCells());
        if (level->getExtraCells() != IntVector(0,0,0))
          levelElem->appendElement("extraCells", level->getExtraCells());
        levelElem->appendElement("anchor", level->getAnchor());
        levelElem->appendElement("id", level->getID());
        if (!level->isStretched()) {
          levelElem->appendElement("cellspacing", level->dCell());
        }
        else {
          for (int axis = 0; axis < 3; axis++) {
            ostringstream axisstr, lowstr, highstr;
            axisstr << axis;
            ProblemSpecP stretch = levelElem->appendChild("StretchPositions");
            stretch->setAttribute("axis", axisstr.str());

            OffsetArray1<double> faces;
            level->getFacePositions((Grid::Axis)axis, faces);
            int low = faces.low();
            int high = faces.high();
            lowstr << low;
            stretch->setAttribute("low", lowstr.str());
            highstr << high;
            stretch->setAttribute("high", highstr.str());
          
            for (int i = low; i < high; i++)
              stretch->appendElement("pos", faces[i]);
          }
        }


        Level::const_patchIterator iter;

        procOnLevel[l].resize(d_myworld->size());

        for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++) {
          const Patch* patch=*iter;
          int proc = lb->getOutputProc(patch);
          procOnLevel[l][proc] = 1;

          Box box = patch->getExtraBox();
          ProblemSpecP patchElem = levelElem->appendChild("Patch");
          patchElem->appendElement("id", patch->getID());
          patchElem->appendElement("proc", proc);
          patchElem->appendElement("lowIndex", patch->getExtraCellLowIndex());
          patchElem->appendElement("highIndex", patch->getExtraCellHighIndex());
          if (patch->getExtraCellLowIndex() != patch->getCellLowIndex())
            patchElem->appendElement("interiorLowIndex", patch->getCellLowIndex());
          if (patch->getExtraCellHighIndex() != patch->getCellHighIndex())
            patchElem->appendElement("interiorHighIndex", patch->getCellHighIndex());
          patchElem->appendElement("nnodes", patch->getNumExtraNodes());
          patchElem->appendElement("lower", box.lower());
          patchElem->appendElement("upper", box.upper());
          patchElem->appendElement("totalCells", patch->getNumExtraCells());
        }
      }
      
      ProblemSpecP dataElem = rootElem->appendChild("Data");
      for(int l=0;l<numLevels;l++) {
        ostringstream lname;
        lname << "l" << l;

        // create a pxxxxx.xml file for each proc doing the outputting
        for(int i=0;i<d_myworld->size();i++) {
          if (i % lb->getNthProc() != 0 || procOnLevel[l][i] == 0)
            continue;
          ostringstream pname;
          pname << lname.str() << "/p" << setw(5) << setfill('0') << i 
                << ".xml";

          ProblemSpecP df = dataElem->appendChild("Datafile");
          df->setAttribute("href",pname.str());
          
          ostringstream procID;
          procID << i;
          df->setAttribute("proc",procID.str());
          
          ostringstream labeltext;
          labeltext << "Processor " << i << " of " << d_myworld->size();
        }
      }
      
      if (hasGlobals) {
        ProblemSpecP df = dataElem->appendChild("Datafile");
        df->setAttribute("href", "global.xml");
      }

      // Add the <Materials> section to the timestep.xml
      SimulationInterface* sim = 
        dynamic_cast<SimulationInterface*>(getPort("sim")); 

      GeometryPieceFactory::resetGeometryPiecesOutput();

      sim->outputProblemSpec(rootElem);

      string name = baseDirs[i]->getName()+"/"+tname.str()+"/timestep.xml";
      rootElem->output(name.c_str());
      // a small convenience to the user who wants to change things when he restarts
      // let him know that some information to change will need to be done in the timestep.xml
      // file instead of the input.xml file.  Only do this once, though.  
      if (firstCheckpointTimestep) {
        // loop over the blocks in timestep.xml and remove them from input.xml, with some exceptions.
        string inputname = d_dir.getName()+"/input.xml";
        ProblemSpecP inputDoc = loadDocument(inputname);
        inputDoc->output((inputname + ".orig").c_str());

        for (ProblemSpecP ps = rootElem->getFirstChild(); ps != 0; ps = ps->getNextSibling()) {
          string nodeName = ps->getNodeName();
          
          if (nodeName == "Meta" || nodeName == "Time" || nodeName == "Grid" || nodeName == "Data") {
            continue;
          }
          
          ProblemSpecP removeNode = inputDoc->findBlock(nodeName);
          if (removeNode != 0) {
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
      
      // copy the component sections of timestep.xml.
      if( d_usingReduceUda ) {
        copy_outputProblemSpec( d_fromDir, d_dir );
      }
    }
  }
  dbg << "  end\n";
}

//______________________________________________________________________
//
void
DataArchiver::scheduleOutputTimestep(vector<DataArchiver::SaveItem>& saveLabels,
                                     const GridP& grid, 
                                     SchedulerP& sched,
                                     bool isThisCheckpoint )
{
  // Schedule a bunch o tasks - one for each variable, for each patch
  int n=0;
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(getPort("load balancer")); 
  
  for(int i=0;i<grid->numLevels();i++) {
    const LevelP& level = grid->getLevel(i);
    vector< SaveItem >::iterator saveIter;
    const PatchSet* patches = lb->getOutputPerProcessorPatchSet(level);
    
    
    string taskName = "DataArchiver::outputVariables";
    if (isThisCheckpoint) {
     taskName += "(checkpoint)";
    }
    
    Task* t = scinew Task(taskName, this, &DataArchiver::outputVariables, isThisCheckpoint?CHECKPOINT:OUTPUT);
    
    //__________________________________
    //
    for(saveIter = saveLabels.begin(); saveIter!= saveLabels.end();
        saveIter++) {
      // check to see if the input file requested to save on this level.
      // check is done by absolute level, or relative to end of levels (-1 finest, -2 second finest,...)
      map<int, MaterialSetP>::iterator iter;

      iter = saveIter->matlSet.find(level->getIndex());
      if (iter == saveIter->matlSet.end())
        iter = saveIter->matlSet.find(level->getIndex() - level->getGrid()->numLevels());
      
      if (iter == saveIter->matlSet.end())
        iter = saveIter->matlSet.find(ALL_LEVELS);
        
      if (iter != saveIter->matlSet.end()) {
        
        const MaterialSubset* matls = iter->second.get_rep()->getUnion();

        // out of domain really is only there to handle the "all-in-one material", but doesn't break anything else
        t->requires(Task::NewDW, (*saveIter).label, matls, Task::OutOfDomain, Ghost::None, 0, true);
        n++;
      }
    }
    t->setType(Task::Output);
    sched->addTask(t, patches, d_sharedState->allMaterials());
  }
  dbg << "  scheduled output task for " << n << " variables\n";
}
//______________________________________________________________________
//
// be sure to call releaseDocument on the value returned
ProblemSpecP DataArchiver::loadDocument(string xmlName)
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

  // add info to index.xml about each global (reduction) var
  // assume for now that global variables that get computed will not
  // change from timestep to timestep
  static bool wereGlobalsAdded = false;
  if (d_writeMeta && !wereGlobalsAdded) {
    wereGlobalsAdded = true;
    // add saved global (reduction) variables to index.xml
    string iname = d_dir.getName()+"/index.xml";
    ProblemSpecP indexDoc = loadDocument(iname);
    
    ProblemSpecP globals = indexDoc->appendChild("globals");
    
    for (vector<SaveItem>::iterator iter = d_saveReductionLabels.begin();
         iter != d_saveReductionLabels.end(); iter++) {
      SaveItem& saveItem = *iter;
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
DataArchiver::outputReductionVars(const ProcessorGroup*,
                              const PatchSubset* /*pss*/,
                              const MaterialSubset* /*matls*/,
                              DataWarehouse* /*old_dw*/,
                              DataWarehouse* new_dw)
{

  if (new_dw->timestepRestarted())
    return;
    
  double start = Time::currentSeconds();
  // Dump the stuff in the reduction saveset into files in the uda
  // at every timestep
  dbg << "  outputReductionVars task begin\n";

  for(int i=0;i<(int)d_saveReductionLabels.size();i++) {
    SaveItem& saveItem = d_saveReductionLabels[i];
    const VarLabel* var = saveItem.label;
    // FIX, see above
    const MaterialSubset* matls = saveItem.getMaterialSet(ALL_LEVELS)->getUnion();
    for (int m = 0; m < matls->size(); m++) {
      int matlIndex = matls->get(m);
      dbg << "    Reduction " << var->getName() << " matl: " << matlIndex << endl;
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
        throw ErrnoException("DataArchiver::outputReduction(): The file \"" + \
              filename.str() + "\" could not be opened for writing!",errno, __FILE__, __LINE__);
      }
      out << setprecision(17) << d_tempElapsedTime << "\t";
      new_dw->print(out, var, 0, matlIndex);
      out << "\n";
    }
  }
  d_sharedState->d_timingStats[SimulationState::OutputTime] += Time::currentSeconds()-start;
  dbg << "  end\n";
}

//______________________________________________________________________
//
void
DataArchiver::outputVariables(const ProcessorGroup * pg,
                              const PatchSubset    * patches,
                              const MaterialSubset * /*matls*/,
                              DataWarehouse        * /*old_dw*/,
                              DataWarehouse        * new_dw,
                              int                    type)
{
  // IMPORTANT - this function should only be called once per processor per level per type
  //   (files will be opened and closed, and those operations are heavy on 
  //   parallel file systems)

  // return if not an outpoint/checkpoint timestep
  if ((!d_isOutputTimestep && type == OUTPUT) || 
      (!d_isCheckpointTimestep && (type == CHECKPOINT || type == CHECKPOINT_REDUCTION))) {
    return;
  }
  dbg << "  outputVariables task begin\n";
  double start = Time::currentSeconds();  
  
#if SCI_ASSERTION_LEVEL >= 2
  // double-check to make sure only called once per level
  int levelid = type != CHECKPOINT_REDUCTION ? getLevel(patches)->getIndex() : -1;
  if (type == OUTPUT) {
    ASSERT(d_outputCalled[levelid] == false);
    d_outputCalled[levelid] = true;
  }
  else if (type == CHECKPOINT) {
    ASSERT(d_checkpointCalled[levelid] == false);
    d_checkpointCalled[levelid] = true;
  }
  else {
    ASSERT(d_checkpointReductionCalled == false);
    d_checkpointReductionCalled = true;
  }
#endif

  vector< SaveItem >& saveLabels = (type == OUTPUT ? d_saveLabels :
                                    type == CHECKPOINT ? d_checkpointLabels : 
                                    d_checkpointReductionLabels);

  //__________________________________
  // debugging output
  // this task should be called once per variable (per patch/matl subset).
  if (dbg.active()) {
    if(type == CHECKPOINT_REDUCTION) {
      dbg << "    reduction";
    } else {
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
  if (type == OUTPUT)
    dir = d_dir;
  else
    dir = d_checkpointsDir;

  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << getTimestepTopLevel();    // could be modified by reduceUda
  
  Dir tdir = dir.getSubdir(tname.str());
  
  string xmlFilename;
  string dataFilebase;
  string dataFilename;
  const Level* level = NULL;

  // find the xml filename and data filename that we will write to
  // Normal reductions will be handled by outputReduction, but checkpoint
  // reductions call this function, and we handle them differently.
  if (type != CHECKPOINT_REDUCTION) {
    // find the level and level number associated with this patch
    ostringstream lname;
    ASSERT(patches->size() != 0);
    ASSERT(patches->get(0) != 0);

    level = patches->get(0)->getLevel();


#if SCI_ASSERTION_LEVEL >= 1
    for(int i=0;i<patches->size();i++)
      ASSERT(patches->get(i)->getLevel() == level);
#endif
    lname << "l" << level->getIndex(); // Hard coded - steve
    Dir ldir = tdir.getSubdir(lname.str());
    
    ostringstream pname;
    pname << "p" << setw(5) << setfill('0') << d_myworld->myrank();
    xmlFilename = ldir.getName() + "/" + pname.str() + ".xml";
    dataFilebase = pname.str() + ".data";
    dataFilename = ldir.getName() + "/" + dataFilebase;
  } else {
    xmlFilename =  tdir.getName() + "/global.xml";
    dataFilebase = "global.data";
    dataFilename = tdir.getName() + "/" + dataFilebase;
  }

  // Not only lock to prevent multiple threads from writing over the same
  // file, but also lock because xerces (DOM..) has thread-safety issues.
  
  int uda_vis_dump = 1;
#if HAVE_PIDX
  uda_vis_dump = 0;
#else
  uda_vis_dump = 1;
#endif
  
  int uda_checkpoint_dump = 0;
  if (type == CHECKPOINT)
    uda_checkpoint_dump = 1;
  
  if (uda_checkpoint_dump == 1 || uda_vis_dump == 1)
  {
    d_outputLock.lock(); 
    { // make sure doc's constructor is called after the lock.
      ProblemSpecP doc; 

      // file-opening flags
      int flags = O_WRONLY|O_CREAT|O_TRUNC;

#if 0
      // DON'T reload a timestep.xml - it will probably mean there was a timestep restart that had written data
    // and we will want to overwrite it
      ifstream test(xmlFilename.c_str());
      if(test) {
        doc = loadDocument(xmlFilename);
      } else
#endif
        doc = ProblemSpec::createDocument("Uintah_Output");

      // Find the end of the file
      ASSERT(doc != 0);
      ProblemSpecP n = doc->findBlock("Variable");
      
      long cur=0;
      while(n != 0) {
        ProblemSpecP endNode = n->findBlock("end");
        ASSERT(endNode != 0);
        long end = atol(endNode->getNodeValue().c_str());
        
        if(end > cur)
          cur=end;
        n = n->findNextBlock("Variable");
      }
      
      //__________________________________
      // Open the data file:
      //
      // Note: At least one time on a BGQ machine (Vulcan@LLNL), with 160K patches, a single checkpoint
      // file failed to open, and it 'crashed' the simulation.  As the other processes on the node 
      // successfully opened their file, it is possible that a second open call would have succeeded.
      // (The original error no was 71.)  Therefore I am using a while loop and counting the 'tries'.
      
      int         tries = 1;
      
      const char* filename = dataFilename.c_str();
      int         fd       = open( filename, flags, 0666 );
      
      while( fd == -1 ) {

        if( tries >= 50 ) {
          ostringstream msg;
          
          msg << "DataArchiver::output(): Failed to open file '" << dataFilename << "' (after 50 tries).";
          cerr << msg.str() << "\n";
          throw ErrnoException( msg.str(), errno, __FILE__, __LINE__ );
        }

        fd = open( filename, flags, 0666 );
        tries++;
      }
      
      if( tries > 1 ) {
        proc0cout << "WARNING: There was a glitch in trying to open the checkpoint file: " 
                  << dataFilename << ". It took " << tries << " tries to successfully open it.";
      }

    //__________________________________
    // loop over variables
      vector<SaveItem>::iterator saveIter;
      for(saveIter = saveLabels.begin(); saveIter!= saveLabels.end(); saveIter++) {
        const VarLabel* var = saveIter->label;
        // check to see if we need to save on this level
        // check is done by absolute level, or relative to end of levels (-1 finest, -2 second finest,...)
        // find the materials to output on that level
        map<int, MaterialSetP>::iterator iter = saveIter->matlSet.end();
        const MaterialSubset* var_matls = 0;
        
        if (level) {
          iter = saveIter->matlSet.find(level->getIndex());
          if (iter == saveIter->matlSet.end())
            iter = saveIter->matlSet.find(level->getIndex() - level->getGrid()->numLevels());
          if (iter == saveIter->matlSet.end())
            iter = saveIter->matlSet.find(ALL_LEVELS);
          if (iter != saveIter->matlSet.end()) {
            var_matls = iter->second.get_rep()->getUnion();
          }
        }
        else { // checkpoint reductions
          map<int, MaterialSetP>::iterator liter;
          for (liter = saveIter->matlSet.begin(); liter != saveIter->matlSet.end(); liter++) {
            var_matls = saveIter->getMaterialSet(liter->first)->getUnion();
            break;
          }
        }
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
        for(int p=0;p<(type==CHECKPOINT_REDUCTION?1:patches->size());p++) {
          const Patch* patch;
          int patchID;
          
          if (type == CHECKPOINT_REDUCTION) {
            // to consolidate into this function, force patch = 0
            patch = 0;
            patchID = -1;
          }
          else {
            patch = patches->get(p);
            patchID = patch->getID();
          }
          
          //__________________________________
          // write info for this variable to current index file
          for(int m=0;m<var_matls->size();m++) {
            
            int matlIndex = var_matls->get(m);
            
            // Variables may not exist when we get here due to something whacky with weird AMR stuff...
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
              cerr << "lseek error - file: " << filename << ", errno=" << errno << '\n';
              throw ErrnoException("DataArchiver::output (lseek call)", errno, __FILE__, __LINE__);
            }
#endif
            // Pad appropriately
            if(cur%PADSIZE != 0) {
              long pad = PADSIZE-cur%PADSIZE;
              char* zero = scinew char[pad];
              memset(zero, 0, pad);
              int err = (int)write(fd, zero, pad);
              if (err != pad) {
                cerr << "Error writing to file: " << filename << ", errno=" << errno << '\n';
                SCI_THROW(ErrnoException("DataArchiver::output (write call)", errno, __FILE__, __LINE__));
              }
              cur+=pad;
              delete[] zero;
            }
            ASSERTEQ(cur%PADSIZE, 0);
            pdElem->appendElement("start", cur);
            
            // output data to data file
            OutputContext oc(fd, filename, cur, pdElem, d_outputDoubleAsFloat && type != CHECKPOINT);
            new_dw->emit(oc, var, matlIndex, patch);
            pdElem->appendElement("end", oc.cur);
            pdElem->appendElement("filename", dataFilebase.c_str());
            
#if SCI_ASSERTION_LEVEL >= 1
            struct stat st;
            int s = fstat(fd, &st);
            
            if(s == -1) {
              cerr << "fstat error - file: " << filename << ", errno=" << errno << '\n';
              throw ErrnoException("DataArchiver::output (stat call)", errno, __FILE__, __LINE__);
            }
            ASSERTEQ(oc.cur, st.st_size);
#endif
            
            cur=oc.cur;
          }
        }
      }
      
      //__________________________________
      // close files and handles 
      int s = close(fd);
      if(s == -1) {
        cerr << "Error closing file: " << filename << ", errno=" << errno << '\n';
        throw ErrnoException("DataArchiver::output (close call)", errno, __FILE__, __LINE__);
      }
      
      doc->output(xmlFilename.c_str());
      //doc->releaseDocument();
      
      d_sharedState->d_timingStats[SimulationState::OutputTime] += Time::currentSeconds()-start;
    }
    d_outputLock.unlock(); 
  }

  dbg << "  end\n";



#if HAVE_PIDX
  //______________________________________________________________________
  //
  bool pidx_io =  true;
  
  if (pidx_io == true && type != CHECKPOINT)
  {
    
    #if 0
    //__________________________________
    //  bulletproofing 
    // Only CCVariables, only Doubles, ints, or Vectors are allowed right now
    for(vector<SaveItem>::iterator iter = saveLabels.begin(); iter!= saveLabels.end(); iter++) {
      
      const VarLabel* label = iter->label;    
      const Uintah::TypeDescription* td = label->typeDescription();
      const Uintah::TypeDescription* subtype = td->getSubType();
      
      // Only CCVariables, only Doubles, ints, or Vectors are allowed
      if(td->getType() != TypeDescription::CCVariable       ||
         ( subtype->getType() != TypeDescription::double_type &&
           subtype->getType() != TypeDescription::int_type    &&
           subtype->getType() != TypeDescription::Vector ) ){
        ostringstream warn;
        warn << "DataArchiver::outputVariables ("<<label->getName() << " " 
             << td->getName() << " ) has not been implemented" << endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    #endif
  
    //__________________________________
    //
    IntVector adjust_offset = IntVector(1,1,1);  //for data conversion, some datasets start at -1,-1,-1, this will soon be handled automatically by pidx 
                                                // <ctc> for temporal jet, set to 0,0,1, use_float=false,agg_size=4
    IntVector clowIndex;
    IntVector chighIndex;
    //level->findIndexRange(lowIndex,highIndex);
    //level->findNodeIndexRange(lowIndex,highIndex);
    //level->findCellIndexRange(lowIndex,highIndex);
    level->findCellIndexRange(clowIndex,chighIndex);
    //printf("clowIndex: %d %d %d, chighindex: %d %d %d\n",clowIndex[0],clowIndex[1],clowIndex[2],chighIndex[0],chighIndex[1],chighIndex[2]);
    //level->findInteriorIndexRange(lowIndex,highIndex);
    //level->findInteriorNodeIndexRange(lowIndex,highIndex);
    //level->findInteriorCellIndexRange(lowIndex,highIndex);
    
    int globalExtents[3];
    globalExtents[0] = chighIndex[0] - clowIndex[0] ;
    globalExtents[1] = chighIndex[1] - clowIndex[1] ;
    globalExtents[2] = chighIndex[2] - clowIndex[2] ;
    
    int number_of_variables;
    int *number_of_materials;
    
    unsigned int timeStep = d_sharedState->getCurrentTopLevelTimeStep();
    
    number_of_variables =  saveLabels.size();
    number_of_materials = (int*)malloc(sizeof(int) * number_of_variables);            // This doesn't look correct  - Todd
    
    string idxFilename(dir.getName());
    idxFilename = idxFilename + ".idx";

    //__________________________________
    //  Debugging
    int rank = pg->myrank();
    if( rank == 0 ) {
      printf("[PIDX] IDX file name = %s\n", (char*)idxFilename.c_str());
      printf("[PIDX] The global volume = %d %d %d\n", globalExtents[0], globalExtents[1], globalExtents[2]);
      printf("[PIDX] Current time step = %d\n", timeStep);
      printf("[PIDX] Total number of variable = %d\n", number_of_variables);
    }
    

    PIDXOutputContext pc;
    // Can this be run in serial without doing a MPI initialize
    pc.initialize(idxFilename, /*timeStep*/PIDX_ts++, globalExtents, d_myworld->getComm());
    
    int vc = 0;
    int vcm = 0;
    int cc_var_count = 0;
    int actual_number_of_variables = 0;
    
    //__________________________________
    //  loop over variables
    //  determine number of 
    vector<SaveItem>::iterator saveIter;
    for(saveIter = saveLabels.begin(); saveIter!= saveLabels.end(); saveIter++) {
      
      const VarLabel* var = saveIter->label;
      string typestr = var->typeDescription()->getName().c_str();
      if (strstr(typestr.c_str(), "CCVariable") == NULL)
        continue;
      
      if (strstr(typestr.c_str(), "Vector") == NULL)
        continue;


      cout << var->getName() << " " << typestr << " me: " << (strstr(typestr.c_str(), "CCVariable") == NULL) << " " << (strstr(typestr.c_str(), "Vector") == NULL) << endl;
      //if (rank == 0)
      //  cout << "type = " << type << endl;
      
      map<int, MaterialSetP>::iterator iter = saveIter->matlSet.end();
      const MaterialSubset* var_matls = 0;

      if (level) {
        iter = saveIter->matlSet.find(level->getIndex());
        
        if (iter == saveIter->matlSet.end()){
          iter = saveIter->matlSet.find(level->getIndex() - level->getGrid()->numLevels());
        }
        if (iter == saveIter->matlSet.end()) {
          iter = saveIter->matlSet.find(ALL_LEVELS);
        }
        if (iter != saveIter->matlSet.end()) {
          var_matls = iter->second.get_rep()->getUnion();
        }
      }
      else { // checkpoint reductions
        map<int, MaterialSetP>::iterator liter;
        for (liter = saveIter->matlSet.begin(); liter != saveIter->matlSet.end(); liter++) {
          var_matls = saveIter->getMaterialSet(liter->first)->getUnion();
          break;
        }
      }
      if (var_matls == 0){
        continue;
      }
      
      number_of_materials[cc_var_count]=var_matls->size();
      //if(rank == 0)
      //  printf("Number of materials for variable %d = %d\n", cc_var_count, var_matls->size());

      cc_var_count++;
      actual_number_of_variables += var_matls->size();
    }
    
    //__________________________________
    //
    int PIDX_debug = 1;
    int PIDX_extra_field = 0;
    if (PIDX_debug == 1){
      PIDX_extra_field = 1;
    }
    
    PIDX_set_variable_count(pc.file, actual_number_of_variables + PIDX_extra_field);
    pc.variable = (PIDX_variable**) malloc(sizeof (PIDX_variable*) * (cc_var_count + PIDX_extra_field));
    memset(pc.variable, 0, sizeof (PIDX_variable*) * cc_var_count);
    
    for(int i = 0 ; i < cc_var_count ; i++) {
      pc.variable[i] = (PIDX_variable*) malloc(sizeof (PIDX_variable) * number_of_materials[i]);
      memset(pc.variable[i], 0, sizeof (PIDX_variable) * number_of_materials[i]);
    }
    
    for(int i = cc_var_count ; i < cc_var_count + PIDX_extra_field ; i++) {
      pc.variable[i] = (PIDX_variable*) malloc(sizeof (PIDX_variable));
      memset(pc.variable[i], 0, sizeof (PIDX_variable));
    }
    
    //__________________________________
    //
#if 1    
    unsigned char ***patch_buffer;
    patch_buffer = (unsigned char***)malloc(sizeof(unsigned char**) * actual_number_of_variables);
    vc = 0;
    for(saveIter = saveLabels.begin(); saveIter!= saveLabels.end(); saveIter++) 
    {
      const VarLabel* var = saveIter->label;
      string type2 = var->typeDescription()->getName().c_str();
      if (strstr(type2.c_str(), "CCVariable") == NULL)
        continue;
      
      if (strstr(type2.c_str(), "Vector") == NULL)
        continue;
      
      int sample_per_variable = 1;
      if (strstr(type2.c_str(), "Vector") != NULL){
        sample_per_variable = 3;
      }
      
      char data_type[512];
      if (strstr(type2.c_str(), "double") != NULL || strstr(type2.c_str(), "Vector") != NULL){
        sprintf(data_type, "%d*float32", sample_per_variable);
      } 
      
      if (strstr(type2.c_str(), "int") != NULL){
        sprintf(data_type, "%d*int32", sample_per_variable); 
      }
      
      //IntVector vhi, vlow, vrange;
      //vlow = var->getCellLowIndex();
      //vhi = var->getCellHighIndex();
      // check to see if we need to save on this level
      // check is done by absolute level, or relative to end of levels (-1 finest, -2 second finest,...)
      // find the materials to output on that level
      
      map<int, MaterialSetP>::iterator iter = saveIter->matlSet.end();
      const MaterialSubset* var_matls = 0;

      if (level) {
        iter = saveIter->matlSet.find(level->getIndex());
        if ( iter == saveIter->matlSet.end() ) {
          iter = saveIter->matlSet.find(level->getIndex() - level->getGrid()->numLevels());
        }
        if ( iter == saveIter->matlSet.end() ){
          iter = saveIter->matlSet.find(ALL_LEVELS);
        }
        if ( iter != saveIter->matlSet.end() ) {
          var_matls = iter->second.get_rep()->getUnion();
        }
      }
      else { // checkpoint reductions
        map<int, MaterialSetP>::iterator liter;
        for (liter = saveIter->matlSet.begin(); liter != saveIter->matlSet.end(); liter++) {
          var_matls = saveIter->getMaterialSet(liter->first)->getUnion();
          break;
        }
      }
      if (var_matls == 0){
        continue;
      }
      
      //__________________________________
      //
      // loop through patches and materials
      for(int m=0;m<var_matls->size();m++){
        int matlIndex = var_matls->get(m);
        string var_mat_name;
        
        if (var_matls->size() == 1) {
          var_mat_name = var->getName();
        } else {
          std::ostringstream s;
          s << m;
          var_mat_name = var->getName() + "_m" + s.str();
        }

        if (strstr(type2.c_str(), "double") != NULL || strstr(type2.c_str(), "Vector") != NULL){
          PIDX_variable_create((char*)var_mat_name.c_str(), sample_per_variable * sizeof(float) * 8, data_type, &(pc.variable[vc][m]));
        
        }else if (strstr(type2.c_str(), "int") != NULL)  { 
          PIDX_variable_create((char*)var_mat_name.c_str(), sample_per_variable * sizeof(int) * 8, data_type, &(pc.variable[vc][m]));
        }
        
        patch_buffer[vcm] = (unsigned char**)malloc(sizeof(unsigned char*) * patches->size());
        
        for(int p=0;p<(type==CHECKPOINT_REDUCTION?1:patches->size());p++)
        {
          const Patch* patch;
          int patchID;
          
          if (type == CHECKPOINT_REDUCTION) {
            // to consolidate into this function, force patch = 0
            patch = 0;
            patchID = -1;
            //patch_buffer[vc][p] = (float*)NULL;

          } else {
            patch = patches->get(p);
            patchID = patch->getID();

            IntVector hiE       = patch->getExtraCellHighIndex();
            IntVector lowE      = patch->getExtraCellLowIndex();
            IntVector nCells_EC = hiE - lowE;
            int totalCells_EC = nCells_EC.x() * nCells_EC.y() * nCells_EC.z();
            
            //if(rank == 0)
            if(vc == 0) {
              printf("[PIDX %d] [%d] Offset and Count %d %d %d : %d %d %d\n", rank, vc, lowE.x(), lowE.y(), lowE.z(), nCells_EC.x(), nCells_EC.y(), nCells_EC.z());
            }
            
            PIDX_point local_offset_point, local_box_count_point;
            
            PIDX_set_point_5D(local_offset_point, lowE.x()+adjust_offset.x(), lowE.y()+adjust_offset.y(), lowE.z()+adjust_offset.z(), 0, 0);
            PIDX_set_point_5D(local_box_count_point, nCells_EC.x(), nCells_EC.y(), nCells_EC.z(), 1, 1);
            
            //if (rank==0)
            //  printf("[PIDX %d] patch_buffer size: %ld\n", rank, sample_per_variable * (use_float?sizeof(float):sizeof(double)) * ((hiE.x() - lowE.x()) * (hiE.y() - lowE.y()) * (hiE.z() - lowE.z())));

            unsigned char *t_buffer = NULL; 
            
            if (strstr(type2.c_str(), "double") != NULL || strstr(type2.c_str(), "Vector") != NULL){
              t_buffer = (unsigned char*)malloc(sample_per_variable * sizeof(double) * totalCells_EC );
              memset(t_buffer, 0, sample_per_variable * sizeof(double) * totalCells_EC );

              patch_buffer[vcm][p] = (unsigned char*)malloc( sample_per_variable * sizeof(float) * totalCells_EC );
              memset(patch_buffer[vcm][p], 0, sample_per_variable * sizeof(float) * totalCells_EC );
            }
            else if (strstr(type2.c_str(), "int") != NULL)
            {
              t_buffer = (unsigned char*)malloc(sample_per_variable * sizeof(int) * totalCells_EC );
              memset(t_buffer, 0, sample_per_variable * sizeof(int) * totalCells_EC );

              patch_buffer[vcm][p] = (unsigned char*)malloc(sample_per_variable * sizeof(int) * totalCells_EC );
              memset(patch_buffer[vcm][p], 0, sample_per_variable * sizeof(int) * totalCells_EC );
            }
            else {
              printf("[1] Error !!!! Unsupported data type");
            }
            
            //__________________________________
            //  Read in Array3 data to t-buffer
            new_dw->emit(pc, var, matlIndex, patch, t_buffer);

            
            if (strstr(type2.c_str(), "double") != NULL || (strstr(type2.c_str(), "Vector") != NULL))
            {
              double* tt_buffer = (double*)malloc(sample_per_variable * sizeof(double) * totalCells_EC);
              memset(tt_buffer, 0, sample_per_variable * sizeof(double) * totalCells_EC);
              memcpy(tt_buffer, t_buffer, sample_per_variable * sizeof(double) * totalCells_EC);
              free(t_buffer);
              
              float* ff_buffer = (float*)malloc(sample_per_variable * sizeof(float) * totalCells_EC);
              memset(ff_buffer, 0, sample_per_variable * sizeof(float) * totalCells_EC);
              std::copy(tt_buffer, tt_buffer + (sample_per_variable * totalCells_EC), ff_buffer);
              free(tt_buffer);
              
              memcpy(patch_buffer[vcm][p], ff_buffer, sample_per_variable * sizeof(float) * totalCells_EC);
              free (ff_buffer);
            }
            else if (strstr(type2.c_str(), "int") != NULL)
            {
              memcpy(patch_buffer[vcm][p], t_buffer, sample_per_variable * sizeof(int) * totalCells_EC);
              free(t_buffer);
            }
            else {
              printf("[2] Error !!!! Unsupported data type");
            }
            
            PIDX_variable_write_data_layout(pc.variable[vc][m], local_offset_point, local_box_count_point, patch_buffer[vcm][p], PIDX_row_major);
                        
          }
        }//  Patches
        PIDX_append_and_write_variable(pc.file, pc.variable[vc][m]);
        vcm++;
        
      }//  Materials
      
      vc++;
    }//  Variables
    
    //______________________________________________________________________
    //      DEBUGGING 
    float **PIDX_patch_buffer;
    if (PIDX_debug == 1)
    {
      PIDX_variable_create("PIDX_test_variable", sizeof(float) * 8, FLOAT32, &(pc.variable[cc_var_count][0]));
      PIDX_patch_buffer = (float**)malloc(sizeof(float*) * patches->size());
      
      for(int p=0;p<(type==CHECKPOINT_REDUCTION?1:patches->size());p++)
      {
        const Patch* patch;

        patch = patches->get(p);
        IntVector hiE       = patch->getExtraCellHighIndex();
        IntVector lowE      = patch->getExtraCellLowIndex();
        IntVector nCells_EC = hiE - lowE;
        int totalCells_EC = nCells_EC.x() * nCells_EC.y() * nCells_EC.z();
            
        printf("[PIDX %d] [%d] Offset and Count %d %d %d : %d %d %d\n", rank, vc, lowE.x(), lowE.y(), lowE.z(), nCells_EC.x(), nCells_EC.y(), nCells_EC.z() );
            
        PIDX_point local_offset_point, local_box_count_point;
            
        PIDX_set_point_5D(local_offset_point, lowE.x()+adjust_offset.x(), lowE.y()+adjust_offset.y(), lowE.z()+adjust_offset.z(), 0, 0);
        PIDX_set_point_5D(local_box_count_point, nCells_EC.x(), nCells_EC.y(), nCells_EC.z(), 1, 1);
            
        PIDX_patch_buffer[p] = (float*)malloc(sizeof(float) * totalCells_EC);
        memset(PIDX_patch_buffer[p], 0, sizeof(float) * totalCells_EC);


        for (int k = 0; k < nCells_EC.z(); k++) {
          for (int j = 0; j < nCells_EC.y(); j++) {
            for (int i = 0; i < nCells_EC.x(); i++) {
              unsigned long long index = (unsigned long long) ( nCells_EC.x() * (nCells_EC.y() * k) + (nCells_EC.x() * j) + i );
              PIDX_patch_buffer[p][index] = 100 + ((globalExtents[0] * globalExtents[1]*((lowE.z()+adjust_offset.z()) + k)) +(globalExtents[0]*((lowE.y()+adjust_offset.y()) + j)) + ((lowE.x()+adjust_offset.x()) + i));
            }
          }
        }
            
        PIDX_variable_write_data_layout(pc.variable[cc_var_count][0], local_offset_point, local_box_count_point, PIDX_patch_buffer[p], PIDX_row_major);
                        
      }//  Patches
      PIDX_append_and_write_variable(pc.file, pc.variable[cc_var_count][0]);
    }  // debugging
    
    
    double start_time, end_time, io_time, max_time;
    start_time = MPI_Wtime();
    PIDX_close(pc.file);
    end_time = MPI_Wtime();
    
#if 1//not PIDX_FLUSH
    //free buffers
    for (int i=0;i<actual_number_of_variables;i++)
    {
      for(int p=0;p<(type==CHECKPOINT_REDUCTION?1:patches->size());p++)
      {
        free(patch_buffer[i][p]);
        patch_buffer[i][p] = 0;
      }
      free(patch_buffer[i]);
      patch_buffer[i] = 0;
    }
    free(patch_buffer);
    patch_buffer = 0;
    
    if (PIDX_debug == 1)
    {
      for(int p=0;p<(type==CHECKPOINT_REDUCTION?1:patches->size());p++)
        free(PIDX_patch_buffer[p]);
      free(PIDX_patch_buffer);
    }
      
#endif

    //free memory
    //for (int i=0; i<number_of_variables; i++)
    for (int i=0; i<cc_var_count + PIDX_extra_field; i++)
      free(pc.variable[i]);
    free(pc.variable); pc.variable=0;
    
    free(number_of_materials); number_of_materials=0;
    

    //compute pidx runtime
    io_time = end_time - start_time;
    MPI_Allreduce(&io_time,&max_time, 1,MPI_DOUBLE,MPI_MAX, d_myworld->getComm() );
    if (io_time == max_time)
      cout << "Timestep = " << timeStep 
           << " Global Volume = " << globalExtents[0] << "," << globalExtents[1]
           << "," << globalExtents[2] << "," 
           << " Throughput = " 
           << (globalExtents[0]*globalExtents[1]*globalExtents[2]*number_of_variables*(sizeof(float)))/(1024.*1024.*max_time) << " MiB/sec " << " Max Time = " << max_time  
           << " Number of variables = " << number_of_variables 
           << endl;
    
#endif
  }
#endif
} // end output()

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
  // a huge memory penalty in one (same penalty if more than) are
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
      cerr << "makeVersionedDir: Error making directory: " << name.str() << "\n";
      throw ErrnoException("DataArchiver.cc: mkdir failed for some "
                           "reason besides dir already exists", errno, __FILE__, __LINE__);
    }
  }

  // if that didn't work, go ahead with the real algorithm

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
          throw ErrnoException("DataArchiver.cc: rmdir failed", errno, __FILE__, __LINE__);
      }
    }
    else {
      if( errno != EEXIST ) {
        cerr << "makeVersionedDir: Error making directory: " << name.str() << "\n";
        throw ErrnoException("DataArchiver.cc: mkdir failed for some "
                             "reason besides dir already exists", errno, __FILE__, __LINE__);
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

  cout << "DataArchiver created " << dirName << endl;
  d_dir = Dir(dirName);
   
} // end makeVersionedDir()


//______________________________________________________________________
//  Determine which labels will be saved.
void
DataArchiver::initSaveLabels(SchedulerP& sched, bool initTimestep)
{
  dbg << "  initSaveLabels()\n";

  // if this is the initTimestep, then don't complain about saving all the vars,
  // just save the ones you can.  They'll most likely be around on the next timestep.

 
  SaveItem saveItem;
  d_saveReductionLabels.clear();
  d_saveLabels.clear();
   
  d_saveLabels.reserve(d_saveLabelNames.size());
  Scheduler::VarLabelMaterialMap* pLabelMatlMap;
  pLabelMatlMap = sched->makeVarLabelMaterialMap();
  

  for (list<SaveNameItem>::iterator it = d_saveLabelNames.begin();
       it != d_saveLabelNames.end(); it++) {

    // go through each of the saveLabelNames we created in problemSetup
    //   see if that variable has been created, set the compression mode
    //   make sure that the scheduler shows that that it has been scheduled
    //   to be computed.  Then save it to saveItems.
    VarLabel* var = VarLabel::find((*it).labelName);    
    
    if (var == NULL) {
      if (initTimestep) {
        continue;
      }else{
        throw ProblemSetupException((*it).labelName +" variable not found to save.", __FILE__, __LINE__);
      }
    }
    
    if ((*it).compressionMode != "") {
      var->setCompressionMode((*it).compressionMode);
    }
      
    Scheduler::VarLabelMaterialMap::iterator found = pLabelMatlMap->find(var->getName());

    if (found == pLabelMatlMap->end()) {
      if (initTimestep) {
        // ignore this on the init timestep, cuz lots of vars aren't computed on the init timestep
        dbg << "    Ignoring var " << it->labelName << " on initialization timestep\n";
        continue;
      } else {
        throw ProblemSetupException((*it).labelName + " variable not computed for saving.", __FILE__, __LINE__);
      }
    }
    saveItem.label = var;
    saveItem.matlSet.clear();
    
    for (ConsecutiveRangeSet::iterator iter = (*it).levels.begin(); iter != (*it).levels.end(); iter++) {

      ConsecutiveRangeSet matlsToSave = (ConsecutiveRangeSet((*found).second)).intersected((*it).matls);
      saveItem.setMaterials(*iter, matlsToSave, d_prevMatls, d_prevMatlSet);

      if (((*it).matls != ConsecutiveRangeSet::all) && ((*it).matls != matlsToSave)) {
        throw ProblemSetupException((*it).labelName + " variable not computed for all materials specified to save.", __FILE__, __LINE__);
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

   for (dep_vector::const_iterator iter = initreqs.begin(); iter != initreqs.end(); iter++) {
     const Task::Dependency* dep = *iter;
     
     ConsecutiveRangeSet levels;
     const PatchSubset* patchSubset = (dep->patches != 0)?
       dep->patches : dep->task->getPatchSet()->getUnion();
     
     for(int i=0;i<patchSubset->size();i++) {
       const Patch* patch = patchSubset->get(i);
       levels.addInOrder(patch->getLevel()->getIndex());
     }

     ConsecutiveRangeSet matls;
     const MaterialSubset* matSubset = (dep->matls != 0) ?
       dep->matls : dep->task->getMaterialSet()->getUnion();
     
     // The matSubset is assumed to be in ascending order or
     // addInOrder will throw an exception.
     matls.addInOrder(matSubset->getVector().begin(),
                      matSubset->getVector().end());

     for(ConsecutiveRangeSet::iterator liter = levels.begin(); liter != levels.end(); liter++) {
       ConsecutiveRangeSet& unionedVarMatls = label_map[dep->var->getName()][*liter];
       unionedVarMatls = unionedVarMatls.unioned(matls);
     }
     
     //cout << "  Adding checkpoint var " << *dep->var << " levels " << levels << " matls " << matls << endl;
   }
         
   d_checkpointLabels.reserve(label_map.size());
   label_type::iterator mapIter;
   bool hasDelT = false;
   for (mapIter = label_map.begin(); mapIter != label_map.end(); mapIter++) {
     VarLabel* var = VarLabel::find(mapIter->first);
     
     if (var == NULL) {
       throw ProblemSetupException(mapIter->first + " variable not found to checkpoint.", __FILE__, __LINE__);
     }
     
     saveItem.label = var;
     saveItem.matlSet.clear();
     map<int, ConsecutiveRangeSet>::iterator liter;
     for (liter = mapIter->second.begin(); liter != mapIter->second.end(); liter++) {
       
       saveItem.setMaterials(liter->first, liter->second, d_prevMatls, d_prevMatlSet);

       if (string(var->getName()) == "delT") {
         hasDelT = true;
       }
     }
     
     // Skip this variable if the default behavior of variable has been overwritten.
     // For example ignore checkpointing PerPatch<FileInfo> variable
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
     if (var == NULL) {
       throw ProblemSetupException("delT variable not found to checkpoint.", __FILE__, __LINE__);
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
  // reuse material sets when the same set of materials is used for different
  // SaveItems in a row -- easier than finding all reusable material set, but
  // effective in many common cases.
  if ((prevMatlSet != 0) && (matls == prevMatls)) {
    matlSet[level] = prevMatlSet;
  }
  else {
    MaterialSetP& m = matlSet[level];
    m = scinew MaterialSet();
    vector<int> matlVec;
    matlVec.reserve(matls.size());
    for (ConsecutiveRangeSet::iterator iter = matls.begin();
         iter != matls.end(); iter++) {
      matlVec.push_back(*iter);
    }
    m->addAll(matlVec);
    prevMatlSet = m;
    prevMatls = matls;
  }
}

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
  if ((d_outputInterval != 0 && time+dt >= d_nextOutputTime) ||
      (d_outputTimestepInterval != 0 && d_currentTimestep+1 > d_nextOutputTimestep)) {
    do_output=true;
    if(!d_wasOutputTimestep)
      recompile=true;
  } else {
    if(d_wasOutputTimestep)
      recompile=true;
  }
  if ((d_checkpointInterval != 0 && time+dt >= d_nextCheckpointTime) ||
      (d_checkpointTimestepInterval != 0 && d_currentTimestep+1 > d_nextCheckpointTimestep) ||
      (d_checkpointWalltimeInterval != 0 && Time::currentSeconds() >= d_nextCheckpointWalltime)) {
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

  for( list<SaveNameItem>::const_iterator it = d_saveLabelNames.begin(); it != d_saveLabelNames.end(); it++ ) {
    if( it->labelName == label ) {
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
    if (Parallel::usingMPI())
      MPI_Barrier(d_myworld->getComm());
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
    if (Parallel::usingMPI())
      MPI_Barrier(d_myworld->getComm());
  }
}

//______________________________________________________________________
//  This will copy the portions of the timestep.xml from the old uda
//  to the new uda.  Specifically, the sections related to the components.
void
DataArchiver::copy_outputProblemSpec( Dir & fromDir, Dir & toDir )
{
  int dir_timestep = getTimestepTopLevel();     // could be modified by reduceUda
  
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

  for (ProblemSpecP ps = inputDoc->getFirstChild(); ps != 0; ps = ps->getNextSibling()) {
    string nodeName = ps->getNodeName();

    if (nodeName == "Meta" || nodeName == "Time" || nodeName == "Grid" || nodeName == "Data") {
      continue;
    }
    cout << "   Now copying the XML node (" << setw(20) << nodeName << ")" << " from: " << fromFile << " to: " << toFile << endl;
    copySection( myFromDir,  myToDir, "timestep.xml", nodeName );
  }
} 

//______________________________________________________________________
// If your using reduceUda then use use a mapping that's defined in reduceUdaSetup()
int
DataArchiver::getTimestepTopLevel()
{
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  
  if ( d_usingReduceUda ) {
    return d_restartTimestepIndicies[timestep];
  }
  else {
    return timestep;
  }
}

//______________________________________________________________________
//
void
DataArchiver::outputTimestep( double time,
                              double delt,
                              const GridP& grid,
                              SchedulerP& sched )
{
  int proc = d_myworld->myrank();

  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(getPort("load balancer")); 

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
  for( int i=0; i<grid->numLevels(); ++i)
  {
    const LevelP& level = grid->getLevel(i);

    const PatchSet* patches = lb->getOutputPerProcessorPatchSet(level);

    outputVariables(NULL, patches->getSubset(proc), NULL, NULL, newDW, OUTPUT);
    outputVariables(NULL, patches->getSubset(proc), NULL, NULL, newDW, CHECKPOINT);
  }

  // Restore the timestep vars so to return to the normal output
  // schedule.
  d_nextOutputTimestep = nextOutputTimestep;
  d_outputTimestepInterval = outputTimestepInterval;

  d_isOutputTimestep = false;
  d_isCheckpointTimestep = false;
}

//______________________________________________________________________
//
void
DataArchiver::checkpointTimestep( double time,
                                  double delt,
                                  const GridP& grid,
                                  SchedulerP& sched )
{
  int proc = d_myworld->myrank();

  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(getPort("load balancer")); 

  DataWarehouse* newDW = sched->getLastDW();

  // Save the vars so to return to the normal output schedule.
  int nextCheckpointTimestep = d_nextCheckpointTimestep;
  int checkpointTimestepInterval = d_checkpointTimestepInterval;

  d_nextCheckpointTimestep = d_sharedState->getCurrentTopLevelTimeStep();;
  d_checkpointTimestepInterval = 1;

  // If needed create checkpoints/index.xml
  if( !d_checkpointsDir.exists())
  {
    if( proc == 0) {
      d_checkpointsDir = d_dir.createSubdir("checkpoints");
      createIndexXML(d_checkpointsDir);
    }
  }

  // Sync up before every rank can use the checkpoints dir
  if (Parallel::usingMPI())
    MPI_Barrier(d_myworld->getComm());

  // Set up the inital bits including the flag d_isCheckpointTimestep
  // which triggers most actions.
  beginOutputTimestep(time, delt, grid);

  // Updaate the main xml file and write the xml file for this
  // timestep.
  writeto_xml_files(delt, grid);

  // For each level get the patches associated with this processor and
  // save the requested output variables.
  for( int i=0; i<grid->numLevels(); ++i)
  {
    const LevelP& level = grid->getLevel(i);

    const PatchSet* patches = lb->getOutputPerProcessorPatchSet(level);

    outputVariables(NULL, patches->getSubset(proc), NULL, NULL, newDW, CHECKPOINT);
  }

  // Restore the vars so to return to the normal output schedule.
  d_nextCheckpointTimestep = nextCheckpointTimestep;
  d_checkpointTimestepInterval = checkpointTimestepInterval;

  d_isOutputTimestep = false;
  d_isCheckpointTimestep = false;
}
