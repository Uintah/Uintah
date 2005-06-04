#include <TauProfilerForSCIRun.h>

#include <Packages/Uintah/CCA/Components/DataArchiver/DataArchiver.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Dataflow/XMLUtil/SimpleErrorHandler.h>

#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Endian.h>
#include <Core/Thread/Time.h>

#include <sgi_stl_warnings_off.h>
#include <iomanip>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <vector>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <strings.h>
#include <unistd.h>
#include <math.h>
#include <sgi_stl_warnings_on.h>

#ifdef _WIN32
#define MAXHOSTNAMELEN 256
#include <winsock2.h>
#endif

#define PADSIZE 1024L

// RNJ - Leave a define that will turn on and off
//       the PVFS fix just in case someone has a
//       problem and needs a way to turn it off.

#define PVFS_FIX

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("DataArchiver", false);

bool DataArchiver::wereSavesAndCheckpointsInitialized = false;

DataArchiver::DataArchiver(const ProcessorGroup* myworld, int udaSuffix)
  : UintahParallelComponent(myworld),
    d_udaSuffix(udaSuffix),
    d_outputLock("DataArchiver output lock")
{
  d_wasOutputTimestep = false;
  d_wasCheckpointTimestep = false;
  d_saveParticleVariables = false;
  d_saveP_x = false;
  //d_currentTime=-1;
  //d_currentTimestep=-1;

  d_XMLIndexDoc = NULL;
  d_CheckpointXMLIndexDoc = NULL;

  d_outputDoubleAsFloat = false;

  d_fileSystemRetrys = 10;
}

DataArchiver::~DataArchiver()
{
}

void DataArchiver::problemSetup(const ProblemSpecP& params,
                                SimulationState* state)
{
   d_sharedState = state;
   ProblemSpecP p = params->findBlock("DataArchiver");

   d_outputDoubleAsFloat = p->findBlock("outputDoubleAsFloat") != 0;

   // set to false if restartSetup is called - we can't do it there
   // as the first timestep doesn't have any tasks
   d_outputInitTimestep = p->findBlock("outputInitTimestep") != 0;
   p->require("filebase", d_filebase);

   // get output timestep or time interval info
   d_outputInterval = 0;
   if(!p->get("outputTimestepInterval", d_outputTimestepInterval))
     d_outputTimestepInterval = 0;
   if(!p->get("outputInterval", d_outputInterval)
      && d_outputTimestepInterval == 0)
     d_outputInterval = 0.0; // default

   if (d_outputInterval != 0.0 && d_outputTimestepInterval != 0)
     throw ProblemSetupException("Use <outputInterval> or <outputTimestepInterval>, not both");
   
   // set default compression mode - can be "tryall", "gzip", "rle", "rle, gzip", "gzip, rle", or "none"
   string defaultCompressionMode = "";
   if (p->get("compression", defaultCompressionMode)) {
     VarLabel::setDefaultCompressionMode(defaultCompressionMode);
   }
   
   // get the variables to save
   map<string, string> attributes;
   SaveNameItem saveItem;
   ProblemSpecP save = p->findBlock("save");
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
	       " indices for saving '" + saveItem.labelName + "'");
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
	       " for saving '" + saveItem.labelName + "'");
      }

      if (saveItem.levels.size() == 0)
	// if materials aren't specified, all valid materials will be saved
	saveItem.levels = ConsecutiveRangeSet::all;
      
      //__________________________________
      //  bullet proofing: must save p.x 
      //  in addition to other particle variables "p.*"
      if (saveItem.labelName == "p.x") {
         d_saveP_x = true;
      }

      string::size_type pos = saveItem.labelName.find("p.");
      if ( pos != string::npos &&  saveItem.labelName != "p.x") {
        d_saveParticleVariables = true;
      }
      
      d_saveLabelNames.push_back(saveItem);
      
      save = save->findNextBlock("save");
   }
   if(d_saveP_x == false && d_saveParticleVariables == true) {
     throw ProblemSetupException(" You must save p.x when saving other particle variables");
   }     
   
   // get checkpoint information
   d_checkpointInterval = 0.0;
   d_checkpointTimestepInterval = 0;
   d_checkpointWalltimeStart = 0;
   d_checkpointWalltimeInterval = 0;
   d_checkpointCycle = 2; /* 2 is the smallest number that is safe
			     (always keeping an older copy for backup) */
                          
   string attrib_1= "", attrib_2= "", attrib_3= "";
   string attrib_4= "", attrib_5= "";
   ProblemSpecP checkpoint = p->findBlock("checkpoint");
   if (checkpoint != 0) {
      attributes.clear();
      checkpoint->getAttributes(attributes);
      
      attrib_1 = attributes["interval"];
      if (attrib_1 != "")
	d_checkpointInterval = atof(attrib_1.c_str());
      
      attrib_2 = attributes["timestepInterval"];
      if (attrib_2 != "")
	d_checkpointTimestepInterval = atoi(attrib_2.c_str());
      
      attrib_3 = attributes["walltimeStart"];
      if (attrib_3 != "")
	d_checkpointWalltimeStart = atoi(attrib_3.c_str());
      
      attrib_4 = attributes["walltimeInterval"];
      if (attrib_4 != "")
	d_checkpointWalltimeInterval = atoi(attrib_4.c_str());
       
      attrib_5 = attributes["cycle"];
      if (attrib_5 != "")
	d_checkpointCycle = atoi(attrib_5.c_str());
   }
   // must specify something
   if ( attrib_1 == "" && attrib_2 == "" && attrib_3 == "" && checkpoint != 0)
     throw ProblemSetupException("ERROR: \n In checkpointing: must specify either interval, timestepInterval or walltimeStart");

   // can't use both checkpointInterval and checkpointTimestepInterval
   if (d_checkpointInterval != 0.0 && d_checkpointTimestepInterval != 0)
     throw ProblemSetupException("Use <checkpoint interval=...> or <checkpoint timestepInterval=...>, not both");

   // can't have a walltimeStart without a walltimeInterval
   if (d_checkpointWalltimeStart != 0 && d_checkpointWalltimeInterval == 0)
     throw ProblemSetupException("<checkpoint walltimeStart must have a corresponding walltimeInterval");

   // set walltimeStart to walltimeInterval if not specified
   if (d_checkpointWalltimeInterval != 0 && d_checkpointWalltimeStart == 0)
     d_checkpointWalltimeStart = d_checkpointWalltimeInterval;
   
   //d_currentTimestep = 0;
   
   int startTimestep;
   if (p->get("startTimestep", startTimestep)) {
     //d_currentTimestep = startTimestep - 1;
   }
  
   d_lastTimestepLocation = "invalid";
   d_wasOutputTimestep = false;

}

void DataArchiver::initializeOutput(const ProblemSpecP& params) {
   if (d_outputInterval == 0.0 && d_outputTimestepInterval == 0 && d_checkpointInterval == 0.0 && d_checkpointTimestepInterval == 0 && d_checkpointWalltimeInterval == 0) 
	return;

   // set up the next output and checkpoint time
   d_nextOutputTime=0.0;
   d_nextOutputTimestep = d_outputInitTimestep?0:1;
   d_nextCheckpointTime=d_checkpointInterval; 
   d_nextCheckpointTimestep=d_checkpointTimestepInterval+1;
   
   if (d_checkpointWalltimeInterval > 0) {
     d_nextCheckpointWalltime=d_checkpointWalltimeStart + 
       (int) Time::currentSeconds();
     if(Parallel::usingMPI()){
       // make sure we are all writing at same time,
       // even if our clocks disagree
       // make decision based on processor zero time
       MPI_Bcast(&d_nextCheckpointWalltime, 1, MPI_INT, 0, d_myworld->getComm());
     }
   } else 
     d_nextCheckpointWalltime=0;

   if(Parallel::usingMPI()){
     // See how many shared filesystems that we have
     double start=Time::currentSeconds();
     string basename;
     if(d_myworld->myrank() == 0){
       // Create a unique string, using hostname+pid
       char* base = strdup(d_filebase.c_str());
       char* p = base+strlen(base);
       for(;p>=base;p--){
	 if(*p == '/'){
	   *++p=0; // keep trailing slash
	   break;
	 }
       }
       if(*p){
	 free(base);
	 base = strdup(".");
       }

       char hostname[MAXHOSTNAMELEN];
       if(gethostname(hostname, MAXHOSTNAMELEN) != 0)
	 strcpy(hostname, "unknown???");
       ostringstream ts;
       ts << base << "-" << hostname << "-" << getpid();

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
       throw ErrnoException("fopen failed for " + fname, errno);
     fprintf(tmpout, "\n");
     if(fflush(tmpout) != 0)
       throw ErrnoException("fflush", errno);
#ifndef _WIN32
     if(fsync(fileno(tmpout)) != 0)
       throw ErrnoException("fsync", errno);
#endif
     if(fclose(tmpout) != 0)
       throw ErrnoException("fclose", errno);
     MPI_Barrier(d_myworld->getComm());
     // See who else we can see
     d_writeMeta=true;
     int i;
     for(i=0;i<d_myworld->myrank();i++){
       ostringstream name;
       name << basename << "-" << i << ".tmp";
       struct stat st;
       int s=stat(name.str().c_str(), &st);
       if(s == 0 && S_ISREG(st.st_mode)){
	 // File exists, we do NOT need to emit metadata
	 d_writeMeta=false;
	 break;
       } else if(errno != ENOENT){
	 cerr << "Cannot stat file: " << name.str() << ", errno=" << errno << '\n';
	 throw ErrnoException("stat", errno);
       }
     }
     MPI_Barrier(d_myworld->getComm());
     if(d_writeMeta){
       makeVersionedDir();
       string fname = myname.str();
       FILE* tmpout = fopen(fname.c_str(), "w");
       if(!tmpout)
	 throw ErrnoException("fopen", errno);
       string dirname = d_dir.getName();
       fprintf(tmpout, "%s\n", dirname.c_str());
       if(fflush(tmpout) != 0)
	 throw ErrnoException("fflush", errno);
#if defined(__APPLE__)
       if(fsync(fileno(tmpout)) != 0)
	 throw ErrnoException("fsync", errno);
#elif !defined(_WIN32)
       if(fdatasync(fileno(tmpout)) != 0)
	 throw ErrnoException("fdatasync", errno);
#endif
       if(fclose(tmpout) != 0)
	 throw ErrnoException("fclose", errno);
     }
     MPI_Barrier(d_myworld->getComm());
     if(!d_writeMeta){
       ostringstream name;
       name << basename << "-" << i << ".tmp";
       ifstream in(name.str().c_str()); 
       if (!in) {
	 throw InternalError("DataArchiver::initializeOutput(): The file \"" + \
			     name.str() + "\" not found on second pass for filesystem discovery!");
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
     if(d_myworld->myrank() == 0){
       double dt=Time::currentSeconds()-start;
       cerr << "Discovered " << nunique << " unique filesystems in " << dt << " seconds\n";
     }
     // Remove the tmp files...
     int s = unlink(myname.str().c_str());
     if(s != 0){
       cerr << "Cannot unlink file: " << myname.str() << '\n';
       throw ErrnoException("unlink", errno);
     }
   } else {
      makeVersionedDir();
      d_writeMeta = true;
   }

   if (d_writeMeta) {
      // create index.xml 
      string inputname = d_dir.getName()+"/input.xml";
      ofstream out(inputname.c_str());
      if (!out) {
	throw ErrnoException("DataArchiver::initializeOutput(): The file \"" + \
			    inputname + "\" could not be opened for writing!",errno);
      }
      out << params << endl; 
      createIndexXML(d_dir);
   
      // create checkpoints/index.xml (if we are saving checkpoints)
      if (d_checkpointInterval != 0.0 || d_checkpointTimestepInterval != 0 ||
	  d_checkpointWalltimeInterval != 0) {
	 d_checkpointsDir = d_dir.createSubdir("checkpoints");
	 createIndexXML(d_checkpointsDir);
      }
   }
   else
      d_checkpointsDir = d_dir.getSubdir("checkpoints");

}


// to be called after problemSetup and initializeOutput get called
void DataArchiver::restartSetup(Dir& restartFromDir, int startTimestep,
				int timestep, double time, bool fromScratch,
				bool removeOldDir)
{
   d_outputInitTimestep = false;
   if (d_writeMeta && !fromScratch) {
      // partial copy of dat files
      copyDatFiles(restartFromDir, d_dir, startTimestep,
		   timestep, removeOldDir);

      copySection(restartFromDir, d_dir, "restarts");
      copySection(restartFromDir, d_dir, "variables");
      copySection(restartFromDir, d_dir, "globals");

      // partial copy of index.xml and timestep directories and
      // similarly for checkpoints
      copyTimesteps(restartFromDir, d_dir, startTimestep, timestep,
		    removeOldDir);
      Dir checkpointsFromDir = restartFromDir.getSubdir("checkpoints");
      bool areCheckpoints = true;
      copyTimesteps(checkpointsFromDir, d_checkpointsDir, startTimestep,
		    timestep, removeOldDir, areCheckpoints);
      copySection(checkpointsFromDir, d_checkpointsDir, "variables");
      copySection(checkpointsFromDir, d_checkpointsDir, "globals");
      if (removeOldDir)
	 restartFromDir.forceRemove(false);
   }
   else if (d_writeMeta) {
     // just add <restart from = ".." timestep = ".."> tag.
     copySection(restartFromDir, d_dir, "restarts");
     string iname = d_dir.getName()+"/index.xml";
     ProblemSpecP indexDoc = loadDocument(iname);
     if (timestep >= 0)
       addRestartStamp(indexDoc, restartFromDir, timestep);
     ofstream copiedIndex(iname.c_str());
     if (!copiedIndex) {
       throw InternalError("DataArchiver::restartSetup(): The file \"" + \
			   iname + "\" could not be opened for writing!");
     }
     copiedIndex << indexDoc << endl;
     indexDoc->releaseDocument();
   }
   
   // set time and timestep variables appropriately
   //d_currentTimestep = timestep;
   
   if (d_outputInterval > 0)
      d_nextOutputTime = d_outputInterval * ceil(time / d_outputInterval);
   else if (d_outputTimestepInterval > 0)
      d_nextOutputTimestep = timestep + d_outputTimestepInterval;
   
   if (d_checkpointInterval > 0)
      d_nextCheckpointTime = d_checkpointInterval * ceil(time / d_checkpointInterval);
   else if (d_checkpointTimestepInterval > 0)
      d_nextCheckpointTimestep = timestep + d_checkpointTimestepInterval;
   if (d_checkpointWalltimeInterval > 0) {
     d_nextCheckpointWalltime = d_checkpointWalltimeInterval + 
       (int)Time::currentSeconds();
     if(Parallel::usingMPI()){
       MPI_Bcast(&d_nextCheckpointWalltime, 1, MPI_INT, 0, d_myworld->getComm());
     }
   }
}

//////////
// Call this when doing a combine_patches run after calling
// problemSetup.  It will copy the data files over and make it ignore
// dumping reduction variables.
void DataArchiver::combinePatchSetup(Dir& fromDir)
{
   if (d_writeMeta) {
     // partial copy of dat files
     copyDatFiles(fromDir, d_dir, 0, -1, false);
     copySection(fromDir, d_dir, "globals");
   }
   
   string iname = fromDir.getName()+"/index.xml";
   ProblemSpecP indexDoc = loadDocument(iname);

   ProblemSpecP globals = indexDoc->findBlock("globals");
   if (globals != 0) {
      ProblemSpecP variable = globals->findBlock("variable");
      while (variable != 0) {
	string varname;
	if (!variable->getAttribute("name", varname))
	  throw InternalError("global variable name attribute not found");

	// this isn't the most efficient, but i don't think it matters
	// to much for this initialization code
	list<SaveNameItem>::iterator it = d_saveLabelNames.begin();
	while (it != d_saveLabelNames.end()) {
	  if ((*it).labelName == varname) {
	    it = d_saveLabelNames.erase(it);
	  }
	  else {
	    it++;
	  }
	}
	variable = variable->findNextBlock("variable");
      }
   }

   // don't transfer checkpoints when combining patches
   d_checkpointInterval = 0.0;
   d_checkpointTimestepInterval = 0;
   d_checkpointWalltimeInterval = 0;

   // output every timestep -- each timestep is transferring data
   d_outputInterval = 0.0;
   d_outputTimestepInterval = 1;

   indexDoc->releaseDocument();
}

void DataArchiver::copySection(Dir& fromDir, Dir& toDir, string section)
{
  // copy chunk labeled section between index.xml files
  string iname = fromDir.getName()+"/index.xml";
  ProblemSpecP indexDoc = loadDocument(iname);

  iname = toDir.getName()+"/index.xml";
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
  
  ofstream indexOut(iname.c_str());
  if (!indexOut) {
    throw InternalError("DataArchiver::copySection(): The file \"" + \
			iname + "\" could not be opened for writing!");
  }
  indexOut << myIndexDoc << endl;

  indexDoc->releaseDocument();
  myIndexDoc->releaseDocument();
}

void DataArchiver::addRestartStamp(ProblemSpecP indexDoc, Dir& fromDir,
				   int timestep)
{
   // add restart history to restarts section
   ProblemSpecP restarts = indexDoc->findBlock("restarts");
   if (restarts == 0) {
     restarts = indexDoc->appendChild("restarts");
   }

   // restart from <dir> at timestep
   ProblemSpecP restartInfo = indexDoc->appendChild("restart",0,1);
   restartInfo->setAttribute("from", fromDir.getName().c_str());
   
   ostringstream timestep_str;
   timestep_str << timestep;

   restartInfo->setAttribute("timestep", timestep_str.str().c_str());   
}

void DataArchiver::copyTimesteps(Dir& fromDir, Dir& toDir, int startTimestep,
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
      if (timestep > startTimestep &&
	  (timestep <= maxTimestep || maxTimestep < 0)) {
	 // copy the timestep directory over
	 map<string,string> attributes;
	 ts->getAttributes(attributes);

	 string hrefNode = attributes["href"];
	 if (hrefNode == "")
	    throw InternalError("timestep href attribute not found");

	 unsigned int href_pos = (unsigned int)hrefNode.find_first_of("/");

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
	 ProblemSpecP newTS = timesteps->appendChild("timestep", 0, 1);

	 ostringstream timestep_str;
	 timestep_str << timestep;

	 newTS->appendText(timestep_str.str().c_str());
	 for (map<string,string>::iterator iter = attributes.begin();
	      iter != attributes.end(); iter++) {
	   newTS->setAttribute((*iter).first, (*iter).second);
	 }
	 
      }
      ts = ts->findNextBlock("timestep");
   }

   // re-output index.xml
   ofstream copiedIndex(iname.c_str());
   if (!copiedIndex) {
     throw InternalError("DataArchiver::copyTimesteps(): The file \"" + \
			 iname + "\" could not be opened for writing!");
   }
   copiedIndex << indexDoc << endl;
   indexDoc->releaseDocument();

   // we don't need the old document anymore...
   oldIndexDoc->releaseDocument();

}

void DataArchiver::copyDatFiles(Dir& fromDir, Dir& toDir, int startTimestep,
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
	    throw InternalError("global variable href attribute not found");
	 const char* href = hrefNode.c_str();

	 ifstream datFile((fromDir.getName()+"/"+href).c_str());
	 if (!datFile) {
	   throw InternalError("DataArchiver::copyDatFiles(): The file \"" + \
			       (fromDir.getName()+"/"+href) + \
			       "\" could not be opened for reading!");
	 }
	 ofstream copyDatFile((toDir.getName()+"/"+href).c_str(), ios::app);
	 if (!copyDatFile) {
	   throw InternalError("DataArchiver::copyDatFiles(): The file \"" + \
			       (toDir.getName()+"/"+href) + \
			       "\" could not be opened for writing!");
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
   indexDoc->releaseDocument();
}

void DataArchiver::createIndexXML(Dir& dir)
{
   ProblemSpecP rootElem = ProblemSpec::createDocument("Uintah_DataArchive");

   rootElem->appendElement("numberOfProcessors", d_myworld->size());

   ProblemSpecP metaElem = rootElem->appendChild("Meta");

   // Some systems dont supply a logname
   const char * logname = getenv("LOGNAME");
   if(logname) metaElem->appendElement("username", logname);

   time_t t = time(NULL) ;
   
   metaElem->appendElement("date", ctime(&t));
   metaElem->appendElement("endianness", endianness().c_str());
   metaElem->appendElement("nBits", (int)sizeof(unsigned long) * 8 );
   
   string iname = dir.getName()+"/index.xml";
   ofstream out(iname.c_str());
   if (!out) {
     throw ErrnoException("DataArchiver::createIndexXML(): The file \"" + \
			 iname + "\" could not be opened for writing!",errno);
   }
   out << rootElem << endl;
   rootElem->releaseDocument();
}

void DataArchiver::finalizeTimestep(double time, double delt,
				    const GridP& grid, SchedulerP& sched,
                                    bool recompile /*=false*/,
                                    int addMaterial /*=0*/)
{
  //this function should get called exactly once per timestep

  static bool wereSavesAndCheckpointsInitialized = false;
  dbg << "DataArchiver finalizeTimestep, delt= " << delt << endl;
  d_tempElapsedTime = time+delt;
  //d_currentTime=time+delt;
  //if (delt != 0)
  //d_currentTimestep++;
  beginOutputTimestep(time, delt, grid);

  // some changes here - we need to redo this if we add a material, or if we schedule output
  // on the initialization timestep (because there will be new computes on subsequent timestep)
  // - BJW
  if (((delt != 0 || d_outputInitTimestep) && !wereSavesAndCheckpointsInitialized) || addMaterial !=0) {
      /* skip the initialization timestep (normally, anyway) for this
         because it needs all computes to be set
         to find the save labels */
    
    if (d_outputInterval != 0.0 || d_outputTimestepInterval != 0) {
      initSaveLabels(sched, delt == 0);
     
      if (!wereSavesAndCheckpointsInitialized)
        indexAddGlobals(); /* add saved global (reduction)
                              variables to index.xml */
    }
    
    // This assumes that the TaskGraph doesn't change after the second
    // timestep and will need to change if the TaskGraph becomes dynamic. 
    //   We also need to do this again if this is the init timestep
    if (delt != 0) {
      wereSavesAndCheckpointsInitialized = true;
    
      // can't do checkpoints on init timestep....
      if (d_checkpointInterval != 0.0 || d_checkpointTimestepInterval != 0 || 
          d_checkpointWalltimeInterval != 0)
        initCheckpoints(sched);
    }
  }

  // we don't want to schedule more tasks unless we're recompiling
  if (!recompile)
    return;
  if ( (d_outputInterval != 0.0 || d_outputTimestepInterval != 0) && 
       (delt != 0 || d_outputInitTimestep)) {
    // Schedule task to dump out reduction variables at every timestep
    Task* t = scinew Task("DataArchiver::outputReduction",
			  this, &DataArchiver::outputReduction);
    
    for(int i=0;i<(int)d_saveReductionLabels.size();i++) {
      SaveItem& saveItem = d_saveReductionLabels[i];
      const VarLabel* var = saveItem.label_;
      const MaterialSubset* matls = saveItem.getMaterialSet()->getUnion();
      t->requires(Task::NewDW, var, matls);
      t->setType(Task::Output);
    }
    
    sched->addTask(t, 0, 0);
    
    dbg << "Created reduction variable output task" << endl;
    if (delt != 0 || d_outputInitTimestep)
      scheduleOutputTimestep(d_dir, d_saveLabels, grid, sched, false);
  }
    
  if (delt != 0) {
    // output checkpoint timestep
    Task* t = scinew Task("DataArchiver::outputCheckpointReduction",
			  this, &DataArchiver::outputCheckpointReduction);
    
    for(int i=0;i<(int)d_checkpointReductionLabels.size();i++) {
      SaveItem& saveItem = d_checkpointReductionLabels[i];
      const VarLabel* var = saveItem.label_;
      const MaterialSubset* matls = saveItem.getMaterialSet()->getUnion();
      t->requires(Task::NewDW, var, matls);
      t->setType(Task::Output);
    }
    sched->addTask(t, 0, 0);
    
    dbg << "Created checkpoint reduction variable output task" << endl;
    
    scheduleOutputTimestep(d_checkpointsDir, d_checkpointLabels,
			   grid, sched, true);
  }
}


void DataArchiver::beginOutputTimestep( double time, double delt,
					const GridP& grid )
{
  // time should be currentTime+delt
  double currentTime = d_sharedState->getElapsedTime();
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  dbg << "beginOutputTimestep called at time=" << currentTime
      << " (" << d_nextOutputTime << "), " << d_outputTimestepInterval 
      << " (" << d_nextOutputTimestep << ")\n";

  // do *not* update d_nextOutputTime or others here.  We need the original
  // values to compare if there is a timestep restart.  See 
  // reEvaluateOutputTimestep
  if (d_outputInterval != 0.0 && (delt != 0 || d_outputInitTimestep)) {
    if(time+delt >= d_nextOutputTime) {
      // output timestep
      d_wasOutputTimestep = true;
      outputTimestep(d_dir, d_saveLabels, time, delt, grid,
		     &d_lastTimestepLocation);
    }
    else {
      d_wasOutputTimestep = false;
    }
  }
  else if (d_outputTimestepInterval != 0 && (delt != 0 || d_outputInitTimestep)) {
    if(timestep >= d_nextOutputTimestep) {
      // output timestep
      d_wasOutputTimestep = true;
      outputTimestep(d_dir, d_saveLabels, time, delt, grid,
		     &d_lastTimestepLocation);
    }
    else {
      d_wasOutputTimestep = false;
    }
  }
  
  int currsecs = (int)Time::currentSeconds();
  if(Parallel::usingMPI() && d_checkpointWalltimeInterval != 0)
     MPI_Bcast(&currsecs, 1, MPI_INT, 0, d_myworld->getComm());
   
  // same thing for checkpoints
  if ((d_checkpointInterval != 0.0 && time+delt >= d_nextCheckpointTime) ||
      (d_checkpointTimestepInterval != 0 &&
       timestep >= d_nextCheckpointTimestep) ||
      (d_checkpointWalltimeInterval != 0 &&
       currsecs >= d_nextCheckpointWalltime)) {
    d_wasCheckpointTimestep=true;
    string timestepDir;
    outputTimestep(d_checkpointsDir, d_checkpointLabels, time, delt,
		   grid, &timestepDir,
		   d_checkpointReductionLabels.size() > 0);
    
    string iname = d_checkpointsDir.getName()+"/index.xml";

    ProblemSpecP index;
    
    if (d_writeMeta) {
      index = loadDocument(iname);
      
      // store a back up in case it dies while writing index.xml
      string ibackup_name = d_checkpointsDir.getName()+"/index_backup.xml";
      ofstream index_backup(ibackup_name.c_str());
      if (!index_backup) {
	throw InternalError("DataArchiver::beginOutputTimestep(): The file \"" + \
			    ibackup_name + "\" could not be opened for writing!");
      }
      index_backup << index << endl;
    }

    d_checkpointTimestepDirs.push_back(timestepDir);
    if ((int)d_checkpointTimestepDirs.size() > d_checkpointCycle) {
      if (d_writeMeta) {
	// remove reference to outdated checkpoint directory from the
	// checkpoint index
	ProblemSpecP ts = index->findBlock("timesteps");
	ProblemSpecP removed;
	do {
	  ProblemSpecP temp = ts->getFirstChild();
	  removed = ts->removeChild(temp);
	} while (removed->getNodeType() != ProblemSpec::ELEMENT_NODE);
	ofstream indexout(iname.c_str());
	if (!indexout) {
	  throw InternalError("DataArchiver::beginOutputTimestep(): The file \"" + \
			      iname + "\" could not be opened for writing!");
	}
	indexout << index << endl;
	
	// remove out-dated checkpoint directory
	Dir expiredDir(d_checkpointTimestepDirs.front());
	expiredDir.forceRemove(false);
      }
      d_checkpointTimestepDirs.pop_front();
    }
    if (d_writeMeta)
      index->releaseDocument();
  } else {
    d_wasCheckpointTimestep=false;
  }
  dbg << "end beginOutputTimestep()\n";
} // end beginOutputTimestep

void DataArchiver::outputTimestep(Dir& baseDir,
				  vector<DataArchiver::SaveItem>&,
				  double /*time*/, double delt,
				  const GridP& grid,
				  string* pTimestepDir /* passed back */,
				  bool /* hasGlobals  = false */)
{
  dbg << "begin outputTimestep()\n";

  int numLevels = grid->numLevels();
  // time should be currentTime+delt
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  
  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << timestep;
  *pTimestepDir = baseDir.getName() + "/" + tname.str();

  // Create the directory for this timestep, if necessary
  if(d_writeMeta){
    Dir tdir;
    try {
      tdir = baseDir.createSubdir(tname.str());
 
    } catch(ErrnoException& e) {
      if(e.getErrno() != EEXIST)
	throw;
      tdir = baseDir.getSubdir(tname.str());
    }
    
    // Create the directory for this level, if necessary
    for(int l=0;l<numLevels;l++){
      ostringstream lname;
      lname << "l" << l;
      Dir ldir;
      try {
	ldir = tdir.createSubdir(lname.str());
      } catch(ErrnoException& e) {
	if(e.getErrno() != EEXIST)
	  throw;
	ldir = tdir.getSubdir(lname.str());
      }
    }
  }
}

void DataArchiver::reEvaluateOutputTimestep(double /*orig_delt*/, double new_delt)
{
  // call this on a timestep restart.  If lowering the delt goes beneath the 
  // threshold, mark it as not an output timestep

  // this is set in finalizeTimestep to time+delt
  d_tempElapsedTime = d_sharedState->getElapsedTime() + new_delt;

  if (d_wasOutputTimestep && d_outputInterval != 0.0 ) {
    if (d_tempElapsedTime < d_nextOutputTime)
      d_wasOutputTimestep = false;
  }
  if (d_wasCheckpointTimestep && d_checkpointInterval != 0.0) {
    if (d_tempElapsedTime < d_nextCheckpointTime) {
      d_wasCheckpointTimestep = false;    
      d_checkpointTimestepDirs.pop_back();
    }
  }
}

      
void
DataArchiver::executedTimestep(double delt, const GridP& grid)
{
  // double time = d_sharedState->getElapsedTime();
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  // if this was an output/checkpoint timestep,
  // determine when the next one will be.

  // don't do this in beginOutputTimestep because the timestep might restart
  // and we need to have the output happen exactly when we need it.
  if (d_wasOutputTimestep) {
    if (d_outputInterval != 0.0) {
      // output timestep
      while (d_tempElapsedTime >= d_nextOutputTime)
	d_nextOutputTime+=d_outputInterval;
    }
    else if (d_outputTimestepInterval != 0) {
      while (timestep >= d_nextOutputTimestep) {
	d_nextOutputTimestep+=d_outputTimestepInterval;
      }
    }
  }

  // to check for output nth proc
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(getPort("load balancer")); 

  if (d_wasCheckpointTimestep) {
    if (d_checkpointInterval != 0.0) {
      while (d_tempElapsedTime >= d_nextCheckpointTime)
	d_nextCheckpointTime += d_checkpointInterval;
    }
    else if (d_checkpointTimestepInterval != 0) {
      while (timestep >= d_nextCheckpointTimestep)
	d_nextCheckpointTimestep += d_checkpointTimestepInterval;
    }
    if (d_checkpointWalltimeInterval != 0) {
      while (Time::currentSeconds() >= d_nextCheckpointWalltime)
	d_nextCheckpointWalltime += d_checkpointWalltimeInterval;
    }
  }


  // start dumping files to disk
  vector<Dir*> baseDirs;
  if (d_wasOutputTimestep)
    baseDirs.push_back(&d_dir);
  if (d_wasCheckpointTimestep)
    baseDirs.push_back(&d_checkpointsDir);

  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << timestep;

  for (int i = 0; i < static_cast<int>(baseDirs.size()); i++) {
    
    // Reference this timestep in index.xml
    if(d_writeMeta){
      string iname = baseDirs[i]->getName()+"/index.xml";

#ifdef PVFS_FIX
      // RNJ - If we already have the XML Index Doc
      //       loaded, don't load it again.

      ProblemSpecP indexDoc;
      bool hasGlobals = false;

      if ( baseDirs[i] == &d_dir ) {
	indexDoc = d_XMLIndexDoc;
	d_XMLIndexDoc = NULL;
      }
      else if ( baseDirs[i] == &d_checkpointsDir ) {
	indexDoc = d_CheckpointXMLIndexDoc;
	d_CheckpointXMLIndexDoc = NULL;
        hasGlobals = d_checkpointReductionLabels.size() > 0;
      }
      else {
	throw "DataArchiver::executedTimestep(): Unknown directory!";
      }
#else
      ProblemSpecP indexDoc = loadDocument(iname);
#endif

      // if this timestep isn't already in index.xml, add it in
      if (indexDoc == 0)
        continue; // output timestep but no variables scheduled to be saved.
      ASSERT(indexDoc != 0);
      ProblemSpecP ts = indexDoc->findBlock("timesteps");
      if(ts == 0){
	ts = indexDoc->appendChild("timesteps");
      }
      bool found=false;
      for(ProblemSpecP n = ts->getFirstChild(); n != 0; n=n->getNextSibling()){
	if(n->getNodeName() == "timestep") {
	  int readtimestep;
	  if(!n->get(readtimestep))
	    throw InternalError("Error parsing timestep number");
	  if(readtimestep == timestep){
	    found=true;
	    break;
	  }
	}
      }
      if(!found){
        // add timestep info
	string timestepindex = tname.str()+"/timestep.xml";      
	
	ProblemSpecP newElem = ts->appendChild("timestep",0,1);
	ostringstream value;
	value << timestep;
	newElem->appendText(value.str().c_str());

	newElem->setAttribute("href", timestepindex.c_str());
	ts->appendText("\n");
      }
      
      ofstream indexOut(iname.c_str());
      if (!indexOut) {
	throw InternalError("DataArchiver::executedTimestep(): The file \"" + \
			    iname + "\" could not be opened for writing!");
      }
      indexOut << indexDoc << endl;
      indexDoc->releaseDocument();

      // make a timestep.xml file for this timestep 
      // we need to do it here in case there is a timestesp restart
      ProblemSpecP rootElem = ProblemSpec::createDocument("Uintah_timestep");

      ProblemSpecP timeElem = rootElem->appendChild("Time");
      timeElem->appendElement("timestepNumber", timestep);
      timeElem->appendElement("currentTime", d_tempElapsedTime);
      timeElem->appendElement("delt", delt);
      
      ProblemSpecP gridElem = rootElem->appendChild("Grid");
      int numLevels = grid->numLevels();
      gridElem->appendElement("numLevels", numLevels);
      for(int l = 0;l<numLevels;l++){
	LevelP level = grid->getLevel(l);
	ProblemSpecP levelElem = gridElem->appendChild("Level", 1, 1);

	if (level->getPeriodicBoundaries() != IntVector(0,0,0))
	  levelElem->appendElement("periodic", level->getPeriodicBoundaries(),0,2);
	levelElem->appendElement("numPatches", level->numPatches(),0,2);
	levelElem->appendElement("totalCells", level->totalCells(),0,2);
        if (level->getExtraCells() != IntVector(0,0,0))
          levelElem->appendElement("extraCells", level->getExtraCells(),0,2);
	levelElem->appendElement("cellspacing", level->dCell(),0,2);
	levelElem->appendElement("anchor", level->getAnchor(),0,2);
	levelElem->appendElement("id", level->getID(),0,2);


	Level::const_patchIterator iter;

	for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
	  const Patch* patch=*iter;
	  Box box = patch->getBox();
	  ProblemSpecP patchElem = levelElem->appendChild("Patch",1,2);
	  patchElem->appendElement("id", patch->getID(),0,3);
	  patchElem->appendElement("lowIndex", patch->getCellLowIndex(),0,3);
	  patchElem->appendElement("highIndex", patch->getCellHighIndex(),0,3);
          if (patch->getCellLowIndex() != patch->getInteriorCellLowIndex())
            patchElem->appendElement("interiorLowIndex", patch->getInteriorCellLowIndex(),0,3);
          if (patch->getCellHighIndex() != patch->getInteriorCellHighIndex())
            patchElem->appendElement("interiorHighIndex", patch->getInteriorCellHighIndex(),0,3);
          patchElem->appendElement("nnodes", patch->getNNodes(),0,3);
	  patchElem->appendElement("lower", box.lower(),0,3);
	  patchElem->appendElement("upper", box.upper(),0,3);
	  patchElem->appendElement("totalCells", patch->totalCells(),0,3);
	}
      }
      
      ProblemSpecP dataElem = rootElem->appendChild("Data");
      for(int l=0;l<numLevels;l++){
	ostringstream lname;
	lname << "l" << l;

        // create a pxxxxx.xml file for each proc doing the outputting
	for(int i=0;i<d_myworld->size();i++){
          if (i % lb->getNthProc() != 0 )
            continue;
	  ostringstream pname;
	  pname << lname.str() << "/p" << setw(5) << setfill('0') << i 
		<< ".xml";

	  ProblemSpecP df = dataElem->appendChild("Datafile",0,1);
	  df->setAttribute("href",pname.str());
	  
	  ostringstream procID;
	  procID << i;
	  df->setAttribute("proc",procID.str());
	  
	  ostringstream labeltext;
	  labeltext << "Processor " << i << " of " << d_myworld->size();

	  df->appendText(labeltext.str().c_str());
	  dataElem->appendText("\n");
	}
      }
      
      if (hasGlobals) {
	ProblemSpecP df = dataElem->appendChild("Datafile",0,1);
	df->setAttribute("href", "global.xml");
	dataElem->appendText("\n");
      }
	 
      string name = baseDirs[i]->getName()+"/"+tname.str()+"/timestep.xml";
      ofstream out(name.c_str());
      if (!out) {
	throw ErrnoException("DataArchiver::outputTimestep(): The file \"" + \
			    name + "\" could not be opened for writing!",errno);
      }
      out << rootElem << endl;
      rootElem->releaseDocument();

    }
  }

#ifdef PVFS_FIX

  d_outputLock.lock(); 
  {
    // Close data file handles used in regular outputs.
      
    map< int, pair<int, char*> >::iterator dataFileHandleIdx = d_DataFileHandles.begin();
    map< int, pair<int, char*> >::iterator dataFileHandleEnd = d_DataFileHandles.end();

    while ( dataFileHandleIdx != dataFileHandleEnd ) {

      int fd = dataFileHandleIdx->second.first;
      char* filename = dataFileHandleIdx->second.second;

      if ( close( fd ) == -1 ) {
	cerr << "Error closing file: " << filename << ", errno=" << errno << '\n';
	throw ErrnoException("DataArchiver::executedTimestep (close call)", errno);
      }

      dataFileHandleIdx++;  
    }

    d_DataFileHandles.clear();


    // Close data file handles used in checkpoint outputs.

    dataFileHandleIdx = d_CheckpointDataFileHandles.begin();
    dataFileHandleEnd = d_CheckpointDataFileHandles.end();
    
    while ( dataFileHandleIdx != dataFileHandleEnd ) {
      
      int fd = dataFileHandleIdx->second.first;
      char* filename = dataFileHandleIdx->second.second;

      if ( close( fd ) == -1 ) {
	cerr << "Error closing file: " << filename << ", errno=" << errno << '\n';
	throw ErrnoException("DataArchiver::executedTimestep (close call)", errno);
      }
      
      dataFileHandleIdx++;  
    }
    
    d_CheckpointDataFileHandles.clear();
    
    
    // Write out XML data files used in regular outputs.

    map< int, ProblemSpecP >::iterator xmlDocIdx = d_XMLDataDocs.begin();
    map< int, ProblemSpecP >::iterator xmlDocEnd = d_XMLDataDocs.end();

    Dir tdir = d_dir.getSubdir(tname.str());

    while ( xmlDocIdx != xmlDocEnd ) {

      ProblemSpecP tempXMLDataFile = xmlDocIdx->second;

      // Get the file name
      
      ostringstream lname;
      lname << "l" << xmlDocIdx->first;
      Dir ldir = tdir.getSubdir(lname.str());        
      ostringstream pname;
      pname << "p" << setw(5) << setfill('0') << d_myworld->myrank();
      string xmlFilename;	
      xmlFilename = ldir.getName() + "/" + pname.str() + ".xml";
	
      // Open the file and write out the XML Doc.
      
      ofstream out(xmlFilename.c_str());
      if (!out) {
	throw ErrnoException("DataArchiver::executedTimestep(): The file \"" + \
			    xmlFilename + "\" could not be opened for writing!",errno);
      }
      out << tempXMLDataFile << endl;
      tempXMLDataFile->releaseDocument();
	
      xmlDocIdx++;

    } // while ( xmlDocIdx != xmlDocEnd ) {

    d_XMLDataDocs.clear();


    // Write out XML data files used in checkpoint outputs.

    xmlDocIdx = d_CheckpointXMLDataDocs.begin();
    xmlDocEnd = d_CheckpointXMLDataDocs.end();

    tdir = d_checkpointsDir.getSubdir(tname.str());

    while ( xmlDocIdx != xmlDocEnd ) {

      ProblemSpecP tempXMLDataFile = xmlDocIdx->second;

      // Get the file name
      
      ostringstream lname;
      lname << "l" << xmlDocIdx->first;
      Dir ldir = tdir.getSubdir(lname.str());        
      ostringstream pname;
      pname << "p" << setw(5) << setfill('0') << d_myworld->myrank();
      string xmlFilename;	
      xmlFilename = ldir.getName() + "/" + pname.str() + ".xml";
	
      // Open the file and write out the XML Doc.
      
      ofstream out(xmlFilename.c_str());
      if (!out) {
	throw ErrnoException("DataArchiver::executedTimestep(): The file \"" + \
              xmlFilename + "\" could not be opened for writing!",errno);
      }
      out << tempXMLDataFile << endl;
      tempXMLDataFile->releaseDocument();
	
      xmlDocIdx++;

    } // while ( xmlDocIdx != xmlDocEnd ) {

    d_CheckpointXMLDataDocs.clear();

  }
  d_outputLock.unlock();

#endif // #ifdef PVFS_FIX

}

void
DataArchiver::scheduleOutputTimestep(Dir& baseDir,
				     vector<DataArchiver::SaveItem>& saveLabels,
				     const GridP& grid, SchedulerP& sched,
				     bool isThisCheckpoint )
{
  // Schedule a bunch o tasks - one for each variable, for each patch
  int n=0;
  for(int i=0;i<grid->numLevels();i++){
    const LevelP& level = grid->getLevel(i);
    vector< SaveItem >::iterator saveIter;
    const PatchSet* patches = level->eachPatch();
    for(saveIter = saveLabels.begin(); saveIter!= saveLabels.end();
	saveIter++) {
      const MaterialSet* matls = (*saveIter).getMaterialSet();
      ConsecutiveRangeSet* range = &saveIter->levels;

      // check to see if the input file requested to save on this level.
      // check is done by absolute level, or relative to end of levels (-1 finest, -2 second finest,...)
      if (range->find(level->getIndex()) != range->end() ||
          range->find(level->getIndex() - level->getGrid()->numLevels()) != range->end()) {
        Task* t = scinew Task((isThisCheckpoint ? "DataArchiver::checkpoint" : "DataArchiver::output"), 
                              this, &DataArchiver::output,
                              &baseDir, (*saveIter).label_, isThisCheckpoint);
        t->requires(Task::NewDW, (*saveIter).label_, Ghost::None);
        t->setType(Task::Output);
        sched->addTask(t, patches, matls);
        n++;
      }
    }
  }
  dbg << "Created " << n << " output tasks\n";
}

// be sure to call releaseDocument on the value returned
ProblemSpecP DataArchiver::loadDocument(string xmlName)
{
  ProblemSpecReader psr(xmlName);
  return psr.readInputFile();
}

const string
DataArchiver::getOutputLocation() const
{
    return d_dir.getName();
}

void DataArchiver::indexAddGlobals()
{
  dbg << "indexAddGlobals()\n";

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
      const VarLabel* var = saveItem.label_;
      const MaterialSubset* matls = saveItem.getMaterialSet()->getUnion();
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

    ofstream indexOut(iname.c_str());
    if (!indexOut) {
      throw InternalError("DataArchiver::indexAddGlobals(): The file \"" + \
	    iname + "\" could not be opened for writing!");
    }
    indexOut << indexDoc << endl;
    indexDoc->releaseDocument();
  }
  dbg << "end indexAddGlobals()\n";
} // end indexAddGlobals()

void DataArchiver::outputReduction(const ProcessorGroup*,
				   const PatchSubset* /*pss*/,
				   const MaterialSubset* /*matls*/,
				   DataWarehouse* /*old_dw*/,
				   DataWarehouse* new_dw)
{
  // Dump the stuff in the reduction saveset into files in the uda
  // at every timestep
  dbg << "DataArchiver::outputReduction called\n";

  for(int i=0;i<(int)d_saveReductionLabels.size();i++) {
    SaveItem& saveItem = d_saveReductionLabels[i];
    const VarLabel* var = saveItem.label_;
    const MaterialSubset* matls = saveItem.getMaterialSet()->getUnion();
    for (int m = 0; m < matls->size(); m++) {
      int matlIndex = matls->get(m);
      dbg << "Reduction matl " << matlIndex << endl;
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
	      filename.str() + "\" could not be opened for writing!",errno);
      }
      out << setprecision(17) << d_tempElapsedTime << "\t";
      new_dw->print(out, var, 0, matlIndex);
      out << "\n";
    }
  }
}

void DataArchiver::outputCheckpointReduction(const ProcessorGroup* world,
					     const PatchSubset*,
					     const MaterialSubset* /*matls*/,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw)
{
  // Dump the stuff in the reduction saveset into files in the uda
  // only on checkpoint timesteps

  if (!d_wasCheckpointTimestep)
    return;
  dbg << "DataArchiver::outputCheckpointReduction called\n";

  for(int i=0;i<(int)d_checkpointReductionLabels.size();i++) {
    SaveItem& saveItem = d_checkpointReductionLabels[i];
    const VarLabel* var = saveItem.label_;
    const MaterialSubset* matls = saveItem.getMaterialSet()->getUnion();
    PatchSubset* patches = scinew PatchSubset(0);
    patches->add(0);
    output(world, patches, matls, old_dw, new_dw, &d_checkpointsDir, var, true);
    delete patches;
  }
}

void DataArchiver::output(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* /*old_dw*/,
			  DataWarehouse* new_dw,
			  Dir* p_dir,
			  const VarLabel* var,
			  bool isThisCheckpoint )
{
  // return if not an outpoint/checkpoint timestep
  if ((!d_wasOutputTimestep && !isThisCheckpoint) || 
      (!d_wasCheckpointTimestep && isThisCheckpoint)) {
    return;
  }
  bool isReduction = var->typeDescription()->isReductionVariable();

  // this task should be called once per variable (per patch/matl subset).
  dbg << "output called ";
  if(patches->size() == 1 && !patches->get(0)){
    dbg << "for reduction";
  } else {
    dbg << "on patches: ";
    for(int p=0;p<patches->size();p++){
      if(p != 0)
	dbg << ", ";
      if (patches->get(p) == 0)
	dbg << -1;
      else
	dbg << patches->get(p)->getID();
    }
  }
  dbg << ", variable: " << var->getName() << ", materials: ";
  for(int m=0;m<matls->size();m++){
    if(m != 0)
      dbg << ", ";
    dbg << matls->get(m);
  }
  dbg << " at time: " << d_sharedState->getCurrentTopLevelTimeStep() << "\n";
  
  ostringstream tname;
  tname << "t" << setw(5) << setfill('0') << d_sharedState->getCurrentTopLevelTimeStep();
  
  Dir tdir = p_dir->getSubdir(tname.str());
  
  string xmlFilename;
  string dataFilebase;
  string dataFilename;
  const Level* level = NULL;

  // find the xml filename and data filename that we will write to
  // Normal reductions will be handled by outputReduction, but checkpoint
  // reductions call this function, and we handle them differently.
  if (!isReduction) {
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

  d_outputLock.lock(); 
  { // make sure doc's constructor is called after the lock.
    ProblemSpecP doc; 

#ifdef PVFS_FIX
    if ( isReduction )
#endif
    {
      ifstream test(xmlFilename.c_str());
      if(test){
	doc = loadDocument(xmlFilename);
      } else {
	doc = ProblemSpec::createDocument("Uintah_Output");
      }
    }
#ifdef PVFS_FIX
    else
    {
      map< int, ProblemSpecP >* currentXMLDataDocMap;
      map< int, ProblemSpecP >::iterator currentXMLDataDoc;

      if ( isThisCheckpoint ) {
	currentXMLDataDocMap = &d_CheckpointXMLDataDocs;
      }
      else {
	currentXMLDataDocMap = &d_XMLDataDocs;
      }
      
      currentXMLDataDoc = currentXMLDataDocMap->find(level->getIndex());

      // RNJ - If we don't have an XML Index Doc, go ahead
      //       and create one, otherwise use the one we
      //       already have.

      if ( currentXMLDataDoc == currentXMLDataDocMap->end() ) {
	doc = ProblemSpec::createDocument("Uintah_Output");
	(*currentXMLDataDocMap)[level->getIndex()] = doc;
      }
      else {
	doc = (*currentXMLDataDocMap)[level->getIndex()];
      }
    }
#endif

    // Find the end of the file
    ASSERT(doc != 0);
    ProblemSpecP n = doc->findBlock("Variable");
    
    long cur=0;
    while(n != 0){
      ProblemSpecP endNode = n->findBlock("end");
      ASSERT(endNode != 0);
      ProblemSpecP tn = endNode->findTextBlock();
      
      long end = atol(tn->getNodeValue().c_str());
      
      if(end > cur)
	cur=end;
      n = n->findNextBlock("Variable");
    }

    int fd;
    char* filename;
#ifdef PVFS_FIX
    if ( isReduction )
#endif
    {
      // Open the data file
      filename = (char*) dataFilename.c_str();
      fd = open(filename, O_WRONLY|O_CREAT, 0666);
      if ( fd == -1 ) {
	cerr << "Cannot open dataFile: " << dataFilename << '\n';
	throw ErrnoException("DataArchiver::output (open call)", errno);
      }
    }
#ifdef PVFS_FIX
    else
    {
      // map of levels to file descriptors.  select between 
      // data files and checkpoint file handles
      map< int, pair<int, char*> >* currentDataFileHandleMap;
      map< int, pair<int, char*> >::iterator currentDataFileHandle;

      if ( isThisCheckpoint ) {
	currentDataFileHandleMap = &d_CheckpointDataFileHandles;
      }
      else {
	currentDataFileHandleMap = &d_DataFileHandles;
      }

      currentDataFileHandle = currentDataFileHandleMap->find(level->getIndex());
      // RNJ - If we haven't created a data file, go ahead
      //       and create one, otherwise use the one we
      //       already have.

      if ( currentDataFileHandle == currentDataFileHandleMap->end() ) {
        filename = (char*) dataFilename.c_str();
	fd = open(filename, O_WRONLY|O_CREAT, 0666);

	if ( fd == -1 ) {
	  cerr << "Cannot open dataFile: " << dataFilename << '\n';
	  throw ErrnoException("DataArchiver::output (open call)", errno);
	}
	else {
	  (*currentDataFileHandleMap)[level->getIndex()] = make_pair(fd, filename);
	}
      }
      else {
	fd = (*currentDataFileHandleMap)[level->getIndex()].first;
        filename = (*currentDataFileHandleMap)[level->getIndex()].second;
      }
    }
#endif    

#if SCI_ASSERTION_LEVEL >= 1
    struct stat st;
    int s = fstat(fd, &st);
    if(s == -1){
      cerr << "Cannot fstat: " << dataFilename << '\n';
      throw ErrnoException("DataArchiver::output (stat call)", errno);
    }
    ASSERTEQ(cur, st.st_size);
#endif
    
    // loop through patches and materials
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      int patchID = patch?patch->getID():-1;
      for(int m=0;m<matls->size();m++){

        // add info for this variable to the current xml file
	int matlIndex = matls->get(m);
	ProblemSpecP pdElem = doc->appendChild("Variable");
	doc->appendText("\n");
	
	pdElem->appendElement("variable", var->getName());
	pdElem->appendElement("index", matlIndex);
	pdElem->appendElement("patch", patchID);
  	pdElem->setAttribute("type",TranslateVariableType( var->typeDescription()->getName().c_str(), isThisCheckpoint ) );
        if (var->getBoundaryLayer() != IntVector(0,0,0))
          pdElem->appendElement("boundaryLayer", var->getBoundaryLayer());
	
#ifdef __sgi
	off64_t ls = lseek64(fd, cur, SEEK_SET);
#else
	off_t ls = lseek(fd, cur, SEEK_SET);
#endif
        if(ls == -1) {
          cerr << "lseek error - file: " << filename << ", errno=" << errno << '\n';
	  throw ErrnoException("DataArchiver::output (lseek call)", errno);
        }
	// Pad appropriately
	if(cur%PADSIZE != 0){
	  long pad = PADSIZE-cur%PADSIZE;
	  char* zero = scinew char[pad];
	  memset(zero, 0, pad);
	  int err = (int)write(fd, zero, pad);
          if (err != pad) {
            cerr << "Error writing to file: " << filename << ", errno=" << errno << '\n';
            SCI_THROW(ErrnoException("DataArchiver::output (write call)", errno));
          }
	  cur+=pad;
	  delete[] zero;
	}
	ASSERTEQ(cur%PADSIZE, 0);
	pdElem->appendElement("start", cur);

        // output data to data file
	OutputContext oc(fd, filename, cur, pdElem, d_outputDoubleAsFloat && !isThisCheckpoint);
	new_dw->emit(oc, var, matlIndex, patch);
	pdElem->appendElement("end", oc.cur);
	pdElem->appendElement("filename", dataFilebase.c_str());

#if SCI_ASSERTION_LEVEL >= 1
	s = fstat(fd, &st);
	if(s == -1) {
          cerr << "fstat error - file: " << filename << ", errno=" << errno << '\n';
	  throw ErrnoException("DataArchiver::output (stat call)", errno);
        }
	ASSERTEQ(oc.cur, st.st_size);
#endif

	cur=oc.cur;
      }
    }

    // close files and handles (with pvfs fix, only do this with reductions).
#ifdef PVFS_FIX
    if ( isReduction )
#endif
    {
      int s = close(fd);
      if(s == -1) {
        cerr << "Error closing file: " << filename << ", errno=" << errno << '\n';
	throw ErrnoException("DataArchiver::output (close call)", errno);
      }
    }

#ifdef PVFS_FIX
    if ( isReduction )
#endif
    {
      ofstream out(xmlFilename.c_str());
      if (!out) {
	throw ErrnoException("DataArchiver::output(): The file \"" + \
	      xmlFilename + "\" could not be opened for writing!",errno);
      }
      out << doc << endl;
      doc->releaseDocument();
    }

    if(d_writeMeta){
      // Rewrite the index if necessary...
      string iname = p_dir->getName()+"/index.xml";
      ProblemSpecP indexDoc;

#ifdef PVFS_FIX
      // grab the corresponding index.xml and open it if necessary
      if ( isThisCheckpoint ) {
	if ( !d_CheckpointXMLIndexDoc ) {
	  d_CheckpointXMLIndexDoc = loadDocument(iname);
	}

	indexDoc = d_CheckpointXMLIndexDoc;
      }
      else {
	if ( !d_XMLIndexDoc ) {
	  d_XMLIndexDoc = loadDocument(iname);
	}

	indexDoc = d_XMLIndexDoc;
      }
#else
      indexDoc = loadDocument(iname);
#endif

      // add variable (as global or variable) to index.xml if not already there.
      ProblemSpecP vs;
      string variableSection = (isReduction) ? "globals" : "variables";
	 
      vs = indexDoc->findBlock(variableSection);
      if(vs == 0){
	vs = indexDoc->appendChild(variableSection.c_str());
      }
      bool found=false;
      for(ProblemSpecP n = vs->getFirstChild(); n != 0; n=n->getNextSibling()){
	if(n->getNodeName() == "variable") {
	  map<string,string> attributes;
	  n->getAttributes(attributes);
	  string varname = attributes["name"];
	  if(varname == "")
	    throw InternalError("varname not found");
	  if(varname == var->getName()){
	    found=true;
	    break;
	  }
	}
      }
      if(!found){
	ProblemSpecP newElem = vs->appendChild("variable");
	newElem->setAttribute("type", TranslateVariableType( var->typeDescription()->getName(), isThisCheckpoint ) );
	newElem->setAttribute("name", var->getName());
	vs->appendText("\n");
      }

#ifndef PVFS_FIX
      ofstream indexOut(iname.c_str());
      if (!indexOut) {
	throw InternalError("DataArchiver::output(): The file \"" + \
	      iname + "\" could not be opened for writing!");
      }
      indexOut << indexDoc << endl;  
      indexDoc->releaseDocument();
#endif

    }
  }
  d_outputLock.unlock(); 

} // end output()

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
  // If there is a real error, then we can throw an exception becuase
  // we don't care about the memory penalty.

  // The lab machines use a script that removes world readable permissions.
  // Create the uda directory setgid csafe so that others in the csafe
  // group can read. Also depends on umask, which is set in sus.cc
#ifndef _WIN32
  const int maxgroups = 20;
  gid_t grouplist[maxgroups];
  int num_groups = getgroups(maxgroups,grouplist);

  // list of csafe gids on the various machines
  const int num_csafe_gids = 3;
  gid_t csafe_groups[num_csafe_gids];
  csafe_groups[0] = 1079;  //sci csafe gid
  csafe_groups[1] = 7545;  //alc uintah gid
  csafe_groups[2] = 49875; //Q uintah gid

  int found_gid = 0;
  gid_t csafe_gid = (gid_t)-1;
  for (int i = 0; i < num_groups && !found_gid; i++){
    for (int j = 0; j < num_csafe_gids && !found_gid; j++){
      if (grouplist[i] == csafe_groups[j]){
        found_gid = 1;
        csafe_gid = grouplist[i];
      }
    }
  }
#endif

  string dirName;

  // first check to see if the suffix passed in on the command line
  // makes a valid uda dir

  if (d_udaSuffix != -1) {
    ostringstream name;
    name << d_filebase << "." << setw(3) << setfill('0') << d_udaSuffix;
    dirName = name.str();
    
    int code = MKDIR( dirName.c_str(), 0777 );
    if( code == 0 ) { // Created the directory successfully
#ifndef _WIN32
      if (chown(dirName.c_str(),(uid_t) -1, (gid_t) csafe_gid) != 0){
	cerr<<"  could not chgrp "<<dirName.c_str()<< " dir to gid "<<csafe_gid<<endl;
	cerr<<strerror(errno)<<endl;
      }
      chmod(dirName.c_str(),0751|S_ISGID);
#endif
      dirCreated = true;
    }
    else if( errno != EEXIST )  {
      cerr << "makeVersionedDir: Error " << errno << " in mkdir\n";
      throw ErrnoException("DataArchiver.cc: mkdir failed for some "
                           "reason besides dir already exists", errno);
    }
  }

  // if that didn't work, go ahead with the real algorithm

  while (!dirCreated) {
    ostringstream name;
    name << d_filebase << "." << setw(3) << setfill('0') << dirNum;
    dirName = name.str();
      
    int code = MKDIR( dirName.c_str(), 0777 );
    if( code == 0 ) // Created the directory successfully
      {
#ifndef _WIN32
        if (chown(dirName.c_str(),(uid_t) -1, (gid_t) csafe_gid) != 0){
          cerr<<"  could not chgrp "<<dirName.c_str()<< " dir to gid "<<csafe_gid<<endl;
	  cerr<<strerror(errno)<<endl;
	}
        chmod(dirName.c_str(),0751|S_ISGID);
#endif
	dirMax = dirNum;
	if (dirMax == dirMin)
	  dirCreated = true;
	else
	  {
	    int code = rmdir( dirName.c_str() );
	    if (code != 0)
	      throw ErrnoException("DataArchiver.cc: rmdir failed", errno);
	  }
      }
    else
      {
	if( errno != EEXIST )
	  {
	    cerr << "makeVersionedDir: Error " << errno << " in mkdir\n";
	    throw ErrnoException("DataArchiver.cc: mkdir failed for some "
				 "reason besides dir already exists", errno);
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
#ifndef _WIN32
  else if ((rc == 0) && (S_ISLNK(sb.st_mode))) {
    unlink(d_filebase.c_str());
    make_link = true;
  }
  if (make_link)
    symlink(dirName.c_str(), d_filebase.c_str());
#endif

  cout << "DataArchiver created " << dirName << endl;
  d_dir = Dir(dirName);
   
}

void  DataArchiver::initSaveLabels(SchedulerP& sched, bool initTimestep)
{
  dbg << "initSaveLabels called\n";

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
    if (var == NULL) 
      throw ProblemSetupException((*it).labelName +
                                  " variable not found to save.");

    if ((*it).compressionMode != "")
      var->setCompressionMode((*it).compressionMode);
      
    Scheduler::VarLabelMaterialMap::iterator found =
      pLabelMatlMap->find(var->getName());

    if (found == pLabelMatlMap->end()) {
      if (initTimestep) {
        // ignore this on the init timestep, cuz lots of vars aren't computed on the init timestep
        dbg << "Ignoring var " << it->labelName << " on initialization timestep\n";
        continue;
      }
      else
        throw ProblemSetupException((*it).labelName +
                                    " variable not computed for saving.");
    }
    saveItem.label_ = var;
    ConsecutiveRangeSet matlsToSave =
      (ConsecutiveRangeSet((*found).second)).intersected((*it).matls);
    saveItem.setMaterials(matlsToSave, prevMatls_, prevMatlSet_);
      
    if (((*it).matls != ConsecutiveRangeSet::all) &&
	((*it).matls != matlsToSave)) {
      throw ProblemSetupException((*it).labelName +
				  " variable not computed for all materials specified to save.");
    }
    saveItem.levels = (*it).levels;
      
    if (saveItem.label_->typeDescription()->isReductionVariable()) {
      d_saveReductionLabels.push_back(saveItem);
    }
    else {
      d_saveLabels.push_back(saveItem);
    }
  }
  //d_saveLabelNames.clear();
  delete pLabelMatlMap;
  dbg << "end of initSaveLabels\n";
}


void DataArchiver::initCheckpoints(SchedulerP& sched)
{
   dbg << "initCheckpoints called\n";
   typedef vector<const Task::Dependency*> dep_vector;
   const dep_vector& initreqs = sched->getInitialRequires();
   SaveItem saveItem;
   d_checkpointReductionLabels.clear();
   d_checkpointLabels.clear();

   map< string, ConsecutiveRangeSet > label_matl_map;

   for (dep_vector::const_iterator iter = initreqs.begin();
	iter != initreqs.end(); iter++) {
      const Task::Dependency* dep = *iter;
      ConsecutiveRangeSet matls;
      const MaterialSubset* matSubset = (dep->matls != 0) ?
	dep->matls : dep->task->getMaterialSet()->getUnion();

      // The matSubset is assumed to be in ascending order or
      // addInOrder will throw an exception.
      matls.addInOrder(matSubset->getVector().begin(),
		       matSubset->getVector().end());
      ConsecutiveRangeSet& unionedVarMatls =
	label_matl_map[dep->var->getName()];
      unionedVarMatls = unionedVarMatls.unioned(matls);      
   }
         
   d_checkpointLabels.reserve(label_matl_map.size());
   map< string, ConsecutiveRangeSet >::iterator mapIter;
   bool hasDelT = false;
   for (mapIter = label_matl_map.begin();
        mapIter != label_matl_map.end(); mapIter++) {
      VarLabel* var = VarLabel::find((*mapIter).first);
      if (var == NULL)
         throw ProblemSetupException((*mapIter).first +
				  " variable not found to checkpoint.");

      saveItem.label_ = var;
      saveItem.setMaterials((*mapIter).second, prevMatls_, prevMatlSet_);
      saveItem.levels = ConsecutiveRangeSet::all;

      if (string(var->getName()) == "delT") {
	hasDelT = true;
      }

      if (saveItem.label_->typeDescription()->isReductionVariable())
         d_checkpointReductionLabels.push_back(saveItem);
      else
         d_checkpointLabels.push_back(saveItem);
   }

   if (!hasDelT) {
     VarLabel* var = VarLabel::find("delT");
     if (var == NULL)
       throw ProblemSetupException("delT variable not found to checkpoint.");
     saveItem.label_ = var;
     ConsecutiveRangeSet globalMatl("-1");
     saveItem.setMaterials(globalMatl, prevMatls_, prevMatlSet_);
     ASSERT(saveItem.label_->typeDescription()->isReductionVariable());
     d_checkpointReductionLabels.push_back(saveItem);
   }     
}

void DataArchiver::SaveItem::setMaterials(const ConsecutiveRangeSet& matls,
					  ConsecutiveRangeSet& prevMatls,
					  MaterialSetP& prevMatlSet)
{
  // reuse material sets when the same set of materials is used for different
  // SaveItems in a row -- easier than finding all reusable material set, but
  // effective in many common cases.
  if ((prevMatlSet != 0) && (matls == prevMatls)) {
    matlSet_ = prevMatlSet;
  }
  else {
    matlSet_ = scinew MaterialSet();
    vector<int> matlVec;
    matlVec.reserve(matls.size());
    for (ConsecutiveRangeSet::iterator iter = matls.begin();
	 iter != matls.end(); iter++) {
      matlVec.push_back(*iter);
    }
    matlSet_->addAll(matlVec);
    prevMatlSet = matlSet_;
    prevMatls = matls;
  }
}

bool DataArchiver::needRecompile(double /*time*/, double /*dt*/,
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
      (d_outputTimestepInterval != 0 && d_currentTimestep+1 > d_nextOutputTimestep)){
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


string DataArchiver::TranslateVariableType( string type, bool isThisCheckpoint )
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
