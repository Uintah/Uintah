#include <Packages/Uintah/CCA/Components/DataArchiver/DataArchiver.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Dataflow/XMLUtil/SimpleErrorHandler.h>
#include <Dataflow/XMLUtil/XMLUtil.h>

#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Thread/Time.h>

#include <iomanip>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <strings.h>
#include <unistd.h>
#include <math.h>
#ifdef __aix
#include <time.h>
#endif

#define PADSIZE 1024L

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static Dir makeVersionedDir(const std::string nameBase);
static DebugStream dbg("DataArchiver", false);
static DOM_Document loadDocument(std::string xmlName);

DataArchiver::DataArchiver(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld),
    d_outputLock("DataArchiver output lock"),
    d_outputReductionLock("DataArchiver reduction output lock")
{
  d_wasOutputTimestep = false;
  d_wasCheckpointTimestep = false;
  d_currentTime=-1;
  d_currentTimestep=-1;
}

DataArchiver::~DataArchiver()
{
}

void DataArchiver::problemSetup(const ProblemSpecP& params)
{
   ProblemSpecP p = params->findBlock("DataArchiver");
   p->require("filebase", d_filebase);
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1.0;

   string defaultCompressionMode = "";
   if (p->get("compression", defaultCompressionMode)) {
     VarLabel::setDefaultCompressionMode(defaultCompressionMode);
   }
   
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

      d_saveLabelNames.push_back(saveItem);
      save = save->findNextBlock("save");
   }
   
   d_checkpointInterval = 0.0;
   d_checkpointCycle = 2; /* 2 is the smallest number that is safe
			     (always keeping an older copy for backup) */
   ProblemSpecP checkpoint = p->findBlock("checkpoint");
   if (checkpoint != 0) {
      attributes.clear();
      checkpoint->getAttributes(attributes);
      string attrib = attributes["interval"];
      if (attrib != "")
	d_checkpointInterval = atof(attrib.c_str());
      attrib = attributes["cycle"];
      if (attrib != "")
	d_checkpointCycle = atoi(attrib.c_str());
   }
   
   d_currentTimestep = 0;
   d_lastTimestepLocation = "invalid";
   d_wasOutputTimestep = false;

   if (d_outputInterval == 0.0 && d_checkpointInterval == 0.0) 
	return;

   d_nextOutputTime=0.0;
   d_nextCheckpointTime=d_checkpointInterval; // no need to checkpoint t=0

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
	   *p=0;
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
       ts << base << "/" << hostname << "-" << getpid();
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
     }
     // Create a file, of the name p0_hostname-p0_pid-processor_number
     ostringstream myname;
     myname << basename << "-" << d_myworld->myrank() << ".tmp";
     {
       // This will be an empty file, everything is encoded in the name anyway
       ofstream tmpout(myname.str().c_str());
     }
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
       d_dir = makeVersionedDir(d_filebase);
       {
	 ofstream tmpout(myname.str().c_str());
	 tmpout << d_dir.getName();
       }
     }
     MPI_Barrier(d_myworld->getComm());
     if(!d_writeMeta){
       ostringstream name;
       name << basename << "-" << i << ".tmp";
       ifstream in(name.str().c_str());
       if(!in){
	 throw InternalError("File not found on second pass for filesystem discovery!");
       }
       string dirname;
       in >> dirname;
       d_dir=Dir(dirname);
     }
     MPI_Barrier(d_myworld->getComm());
     // Remove the tmp files...
     int s = unlink(myname.str().c_str());
     if(s != 0){
       cerr << "Cannot unlink file: " << myname.str() << '\n';
       throw ErrnoException("unlink", errno);
     }
     int count=d_writeMeta?1:0;
     int nunique;
     MPI_Reduce(&count, &nunique, 1, MPI_INT, MPI_SUM, 0,
		d_myworld->getComm());
     if(d_myworld->myrank() == 0){
       double dt=Time::currentSeconds()-start;
       cerr << "Discovered " << nunique << " unique filesystems in " << dt << " seconds\n";
     }
   } else {
      d_dir = makeVersionedDir(d_filebase);
      d_writeMeta = true;
   }

   if (d_writeMeta) {
      string inputname = d_dir.getName()+"/input.xml";
      ofstream out(inputname.c_str());
      out << params->getNode().getOwnerDocument() << endl;
      createIndexXML(d_dir);
      
      if (d_checkpointInterval != 0.0) {
	 d_checkpointsDir = d_dir.createSubdir("checkpoints");
	 createIndexXML(d_checkpointsDir);
      }
   }
   else
      d_checkpointsDir = d_dir.getSubdir("checkpoints");
}

// to be called after problemSetup gets called
void DataArchiver::restartSetup(Dir& restartFromDir, int startTimestep,
				int timestep, double time, bool fromScratch,
				bool removeOldDir)
{
   if (d_writeMeta && !fromScratch) {
      // partial copy of dat files
      copyDatFiles(restartFromDir, d_dir, startTimestep,
		   timestep, removeOldDir);

      copySection(restartFromDir, "restarts");

      // partial copy of index.xml and timestep directories and
      // similarly for checkpoints
      copyTimesteps(restartFromDir, d_dir, startTimestep, timestep,
		    removeOldDir);
      Dir checkpointsFromDir = restartFromDir.getSubdir("checkpoints");
      bool areCheckpoints = true;
      copyTimesteps(checkpointsFromDir, d_checkpointsDir, startTimestep,
		    timestep, removeOldDir, areCheckpoints);
      if (removeOldDir)
	 restartFromDir.forceRemove();
   }
   else if (d_writeMeta) {
     // just add <restart from = ".." timestep = ".."> tag.
     copySection(restartFromDir, "restarts");
     string iname = d_dir.getName()+"/index.xml";
     DOM_Document indexDoc = loadDocument(iname);
     if (timestep >= 0)
       addRestartStamp(indexDoc, restartFromDir, timestep);
     ofstream copiedIndex(iname.c_str());
     copiedIndex << indexDoc << endl;
   }
   
   // set time and timestep variables appropriately
   d_currentTimestep = timestep;
   
   if (d_outputInterval > 0)
      d_nextOutputTime = d_outputInterval * ceil(time / d_outputInterval);

   if (d_checkpointInterval > 0)
      d_nextCheckpointTime = d_checkpointInterval *
	 ceil(time / d_checkpointInterval);
}

void DataArchiver::copySection(Dir& fromDir, string section)
{
  string iname = fromDir.getName()+"/index.xml";
  DOM_Document indexDoc = loadDocument(iname);

  iname = d_dir.getName()+"/index.xml";
  DOM_Document myIndexDoc = loadDocument(iname);
  
  DOM_Node sectionNode = findNode(section, indexDoc.getDocumentElement());
  if (sectionNode != 0) {
    DOM_Node newNode = myIndexDoc.importNode(sectionNode, true);
    
    // replace whatever was in the section previously
    DOM_Node mySectionNode = findNode(section, myIndexDoc.getDocumentElement());
    if (mySectionNode != 0) {
      myIndexDoc.getDocumentElement().replaceChild(newNode, mySectionNode);
    }
    else {
      myIndexDoc.getDocumentElement().appendChild(myIndexDoc.createTextNode("\n"));
      myIndexDoc.getDocumentElement().appendChild(newNode);
    }
  }
  
  ofstream indexOut(iname.c_str());
  indexOut << myIndexDoc << endl;
}

void DataArchiver::addRestartStamp(DOM_Document indexDoc, Dir& fromDir,
				   int timestep)
{
   DOM_Text leader = indexDoc.createTextNode("\n");
   DOM_Text returnTab = indexDoc.createTextNode("\n\t");
   DOM_Node restarts = findNode("restarts", indexDoc.getDocumentElement());
   if (restarts == 0) {
     restarts = indexDoc.createElement("restarts");
     indexDoc.getDocumentElement().appendChild(leader);
     indexDoc.getDocumentElement().appendChild(restarts);
   }
   DOM_Element restartInfo = indexDoc.createElement("restart");
   restartInfo.setAttribute("from", fromDir.getName().c_str());
   ostringstream timestep_str;
   timestep_str << timestep;
   restartInfo.setAttribute("timestep", timestep_str.str().c_str());
   restarts.appendChild(returnTab);
   restarts.appendChild(restartInfo);
   restarts.appendChild(leader);
}

void DataArchiver::copyTimesteps(Dir& fromDir, Dir& toDir, int startTimestep,
				 int maxTimestep, bool removeOld,
				 bool areCheckpoints /*=false*/)
{
   string old_iname = fromDir.getName()+"/index.xml";
   DOM_Document oldIndexDoc = loadDocument(old_iname);
   string iname = toDir.getName()+"/index.xml";
   DOM_Document indexDoc = loadDocument(iname);
   DOM_Text leader = indexDoc.createTextNode("\n");
   DOM_Node oldTimesteps = findNode("timesteps",
				    oldIndexDoc.getDocumentElement());
   DOM_Node ts;
   if (oldTimesteps != 0)
      ts = findNode("timestep", oldTimesteps);

   // while we're at it, add restart information to index.xml
   if (maxTimestep >= 0)
     addRestartStamp(indexDoc, fromDir, maxTimestep);

   // create timesteps element if necessary
   DOM_Node timesteps = findNode("timesteps",
				 indexDoc.getDocumentElement());
   if (timesteps == 0) {
      leader = indexDoc.createTextNode("\n");
      indexDoc.getDocumentElement().appendChild(leader);
      timesteps = indexDoc.createElement("timesteps");
      indexDoc.getDocumentElement().appendChild(timesteps);
   }
   
   int timestep;
   while (ts != 0) {
      get(ts, timestep);
      if (timestep > startTimestep &&
	  (timestep <= maxTimestep || maxTimestep < 0)) {
	 // copy the timestep directory over
	 DOM_NamedNodeMap attributes = ts.getAttributes();
	 DOM_Node hrefNode = attributes.getNamedItem("href");
	 if (hrefNode == 0)
	    throw InternalError("timestep href attribute not found");
	 char* href = hrefNode.getNodeValue().transcode();
	 strtok(href, "/"); // just grab the directory part
	 Dir timestepDir = fromDir.getSubdir(href);
	 if (removeOld)
	    timestepDir.move(toDir);
	 else
	    timestepDir.copy(toDir);

	 if (areCheckpoints)
	    d_checkpointTimestepDirs.push_back(toDir.getSubdir(href).getName());
	 
	 delete[] href;

	 // add the timestep to the index.xml
	 leader = indexDoc.createTextNode("\n\t");
	 timesteps.appendChild(leader);
	 DOM_Element newTS = indexDoc.createElement("timestep");
	 ostringstream timestep_str;
	 timestep_str << timestep;
	 DOM_Text value = indexDoc.createTextNode(timestep_str.str().c_str());
	 newTS.appendChild(value);
	 for (unsigned int i = 0; i < attributes.getLength(); i++)
	    newTS.setAttribute(attributes.item(i).getNodeName(),
			       attributes.item(i).getNodeValue());
	 
	 timesteps.appendChild(newTS); // copy to new index.xml
	 DOM_Text trailer = indexDoc.createTextNode("\n");
	 timesteps.appendChild(trailer);
      }
      ts = findNextNode("timestep", ts);
   }

   ofstream copiedIndex(iname.c_str());
   copiedIndex << indexDoc << endl;
}

void DataArchiver::copyDatFiles(Dir& fromDir, Dir& toDir, int startTimestep,
				int maxTimestep, bool removeOld)
{
   char buffer[1000];

   // find the dat file via the globals block in index.xml
   string iname = fromDir.getName()+"/index.xml";
   DOM_Document indexDoc = loadDocument(iname);

   DOM_Node globals = findNode("globals", indexDoc.getDocumentElement());
   if (globals != 0) {
      DOM_Node variable = findNode("variable", globals);
      while (variable != 0) {
	 DOM_NamedNodeMap attributes = variable.getAttributes();
	 DOM_Node hrefNode = attributes.getNamedItem("href");
	 if (hrefNode == 0)
	    throw InternalError("global variable href attribute not found");
	 char* href = hrefNode.getNodeValue().transcode();
	 
	 // copy up to maxTimestep lines of the old dat file to the copy
	 ifstream datFile((fromDir.getName()+"/"+href).c_str());
	 ofstream copyDatFile((toDir.getName()+"/"+href).c_str(), ios::app);
	 int timestep = startTimestep;
	 while (datFile.getline(buffer, 1000) &&
		(timestep < maxTimestep || maxTimestep < 0)) {
	    copyDatFile << buffer << endl;
	    timestep++;
	 }
	 datFile.close();

	 if (removeOld) 
	    fromDir.remove(href);
	 
	 delete[] href;
	 
	 variable = findNextNode("variable", variable);
      }
   }
}

void DataArchiver::createIndexXML(Dir& dir)
{
   DOM_DOMImplementation impl;
    
   DOM_Document doc = impl.createDocument(0,"Uintah_DataArchive",
					     DOM_DocumentType());
   DOM_Element rootElem = doc.getDocumentElement();

   appendElement(rootElem, "numberOfProcessors", d_myworld->size());

   DOM_Element metaElem = doc.createElement("Meta");
   rootElem.appendChild(metaElem);
   appendElement(metaElem, "username", getenv("LOGNAME"));
   time_t t = time(NULL) ;
   appendElement(metaElem, "date", ctime(&t));
   
   string iname = dir.getName()+"/index.xml";
   ofstream out(iname.c_str());
   out << doc << endl;
}

void DataArchiver::finalizeTimestep(double time, double delt,
				    const LevelP& level, SchedulerP& sched)
{
  static bool wereSavesAndCheckpointsInitialized = false;
  cout << "Finalize delt= " << delt << endl;
  d_currentTime=time+delt;

  if (!wereSavesAndCheckpointsInitialized &&
      !(delt == 0) /* skip the initialization timestep for this
		      because it needs all computes to be set
		      to find the save labels */) {
    
    // This assumes that the TaskGraph doesn't change after the second
    // timestep and will need to change if the TaskGraph becomes dynamic. 
    wereSavesAndCheckpointsInitialized = true;
    
    if (d_outputInterval != 0.0) {
      initSaveLabels(sched);
      indexAddGlobals(); /* add saved global (reduction)
			    variables to index.xml */
    }

    if (d_checkpointInterval != 0)
      initCheckpoints(sched);
  }
  
  bool do_output=false;
  if (d_outputInterval != 0.0 && delt != 0) {
    // Schedule task to dump out reduction variables at every timestep
    Task* t = scinew Task("DataArchiver::outputReduction",
			  this, &DataArchiver::outputReduction);
    
    for(int i=0;i<(int)d_saveReductionLabels.size();i++) {
      SaveItem& saveItem = d_saveReductionLabels[i];
      const VarLabel* var = saveItem.label_;
      const MaterialSubset* matls = saveItem.getMaterialSet()->getUnion();
      t->requires(Task::NewDW, var, matls);
    }
    
    sched->addTask(t, 0, 0);
    
    dbg << "Created reduction variable output task" << endl;
    if (d_wasOutputTimestep){
      scheduleOutputTimestep(d_dir, d_saveLabels, level, sched);
      do_output=true;
    }
  }
    
  if (d_wasCheckpointTimestep){
    // output checkpoint timestep
    Task* t = scinew Task("DataArchiver::outputCheckpointReduction",
			  this, &DataArchiver::outputCheckpointReduction);
    
    for(int i=0;i<(int)d_checkpointReductionLabels.size();i++) {
      SaveItem& saveItem = d_checkpointReductionLabels[i];
      const VarLabel* var = saveItem.label_;
      const MaterialSubset* matls = saveItem.getMaterialSet()->getUnion();
      t->requires(Task::NewDW, var, matls);
    }
    sched->addTask(t, 0, 0);
    
    dbg << "Created checkpoint reduction variable output task" << endl;

    scheduleOutputTimestep(d_checkpointsDir, d_checkpointLabels,
			   level, sched);
    do_output=true;
  }
  if(do_output && delt == 0)
    beginOutputTimestep(time, delt, level);
}


void DataArchiver::beginOutputTimestep(double time, double delt,
				       const LevelP& level)
{
  d_currentTime=time+delt;
  dbg << "beginOutputTimestep called at time=" << d_currentTime << '\n';
  if (d_outputInterval != 0.0 && delt != 0) {
    if(d_currentTime >= d_nextOutputTime) {
      // output timestep
      d_wasOutputTimestep = true;
      outputTimestep(d_dir, d_saveLabels, time, delt, level,
		     &d_lastTimestepLocation);
      while (d_currentTime >= d_nextOutputTime)
	d_nextOutputTime+=d_outputInterval;
    }
    else
      d_wasOutputTimestep = false;
  }
  
  if (d_checkpointInterval != 0 && d_currentTime >= d_nextCheckpointTime) {
    d_wasCheckpointTimestep=true;
    string timestepDir;
    outputTimestep(d_checkpointsDir, d_checkpointLabels, time, delt,
		   level, &timestepDir,
		   d_checkpointReductionLabels.size() > 0);
    
    string iname = d_checkpointsDir.getName()+"/index.xml";
    DOM_Document index;
    
    if (d_writeMeta) {
      index = loadDocument(iname);
      
      // store a back up in case it dies while writing index.xml
      string ibackup_name = d_checkpointsDir.getName()+"/index_backup.xml";
      ofstream index_backup(ibackup_name.c_str());
      index_backup << index << endl;
    }
      
    d_checkpointTimestepDirs.push_back(timestepDir);
    if ((int)d_checkpointTimestepDirs.size() > d_checkpointCycle) {
      if (d_writeMeta) {
	// remove reference to outdated checkpoint directory from the
	// checkpoint index
	DOM_Node ts = findNode("timesteps", index.getDocumentElement());
	DOM_Node removed;
	do {
	  removed = ts.removeChild(ts.getFirstChild());
	} while (removed.getNodeType() != DOM_Node::ELEMENT_NODE);
	ofstream indexout(iname.c_str());
	indexout << index << endl;
	
	// remove out-dated checkpoint directory
	Dir expiredDir(d_checkpointTimestepDirs.front());
	expiredDir.forceRemove();
      }
      d_checkpointTimestepDirs.pop_front();
    }
    while (d_currentTime >= d_nextCheckpointTime)
      d_nextCheckpointTime += d_checkpointInterval;
  } else {
    d_wasCheckpointTimestep=false;
  }
}

void DataArchiver::outputTimestep(Dir& baseDir,
				  vector<DataArchiver::SaveItem>&,
				  double time, double delt,
				  const LevelP& level,
				  string* pTimestepDir /* passed back */,
				  bool hasGlobals /* = false */)
{
  int timestep = d_currentTimestep;
  
  ostringstream tname;
  tname << "t" << setw(4) << setfill('0') << timestep;
  *pTimestepDir = baseDir.getName() + "/" + tname.str();

  // Create the directory for this timestep, if necessary
  if(d_writeMeta){
    Dir tdir;
    try {
      tdir = baseDir.createSubdir(tname.str());
      
      DOM_DOMImplementation impl;
	 
      DOM_Document doc = impl.createDocument(0,"Uintah_timestep",
					     DOM_DocumentType());
      DOM_Element rootElem = doc.getDocumentElement();

      DOM_Element timeElem = doc.createElement("Time");
      rootElem.appendChild(timeElem);
      
      appendElement(timeElem, "timestepNumber", timestep);
      appendElement(timeElem, "currentTime", time+delt);
      appendElement(timeElem, "delt", delt);
      
      DOM_Element gridElem = doc.createElement("Grid");
      rootElem.appendChild(gridElem);
      
      GridP grid = level->getGrid();
      int numLevels = grid->numLevels();
      appendElement(gridElem, "numLevels", numLevels);
      for(int l = 0;l<numLevels;l++){
	LevelP level = grid->getLevel(l);
	DOM_Element levelElem = doc.createElement("Level");
	gridElem.appendChild(levelElem);
	
	appendElement(levelElem, "numPatches", level->numPatches());
	appendElement(levelElem, "totalCells", level->totalCells());
	appendElement(levelElem, "cellspacing", level->dCell());
	appendElement(levelElem, "anchor", level->getAnchor());
	Level::const_patchIterator iter;
	for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
	  const Patch* patch=*iter;
	  DOM_Element patchElem = doc.createElement("Patch");
	  levelElem.appendChild(patchElem);
	  appendElement(patchElem, "id", patch->getID());
	  appendElement(patchElem, "lowIndex", patch->getCellLowIndex());
	  appendElement(patchElem, "highIndex", patch->getCellHighIndex());
	  Box box = patch->getBox();
	  appendElement(patchElem, "lower", box.lower());
	  appendElement(patchElem, "upper", box.upper());
	  appendElement(patchElem, "totalCells", patch->totalCells());
	}
      }
      DOM_Element dataElem = doc.createElement("Data");
      rootElem.appendChild(dataElem);
      ostringstream lname;
      {
	lname << "l0"; // Hard coded - steve
	for(int i=0;i<d_myworld->size();i++){
	  ostringstream pname;
	  pname << lname.str() << "/p" << setw(5) << setfill('0') << i << ".xml";
	  DOM_Text leader = doc.createTextNode("\n\t");
	  dataElem.appendChild(leader);
	  DOM_Element df = doc.createElement("Datafile");
	  dataElem.appendChild(df);
	  df.setAttribute("href", pname.str().c_str());
	  ostringstream procID;
	  procID << i;
	  df.setAttribute("proc", procID.str().c_str());
	  ostringstream labeltext;
	  labeltext << "Processor " << i << " of " << d_myworld->size();
	  DOM_Text label = doc.createTextNode(labeltext.str().c_str());
	  df.appendChild(label);
	  DOM_Text trailer = doc.createTextNode("\n");
	  dataElem.appendChild(trailer);
	}
      }
	 
      if (hasGlobals) {
	DOM_Text leader = doc.createTextNode("\n\t");
	dataElem.appendChild(leader);
	DOM_Element df = doc.createElement("Datafile");
	dataElem.appendChild(df);
	df.setAttribute("href", "global.xml");
	DOM_Text trailer = doc.createTextNode("\n");
	dataElem.appendChild(trailer);
      }
	 
      string name = tdir.getName()+"/timestep.xml";
      ofstream out(name.c_str());
      out << doc << endl;
      
    } catch(ErrnoException& e) {
      if(e.getErrno() != EEXIST)
	throw;
      tdir = baseDir.getSubdir(tname.str());
    }
      
    // Create the directory for this level, if necessary
    ostringstream lname;
    lname << "l0"; // Hard coded - steve
    Dir ldir;
    try {
      ldir = tdir.createSubdir(lname.str());
    } catch(ErrnoException& e) {
      if(e.getErrno() != EEXIST)
	throw;
      ldir = tdir.getSubdir(lname.str());
    }
      
    // Reference this timestep in index.xml
    string iname = baseDir.getName()+"/index.xml";
    DOM_Document indexDoc = loadDocument(iname);
    DOM_Node ts = findNode("timesteps", indexDoc.getDocumentElement());
    if(ts == 0){
      DOM_Text leader = indexDoc.createTextNode("\n");
      indexDoc.getDocumentElement().appendChild(leader);
      ts = indexDoc.createElement("timesteps");
      indexDoc.getDocumentElement().appendChild(ts);
    }
    bool found=false;
    for(DOM_Node n = ts.getFirstChild(); n != 0; n=n.getNextSibling()){
      if(n.getNodeName().equals(DOMString("timestep"))){
	int readtimestep;
	if(!get(n, readtimestep))
	  throw InternalError("Error parsing timestep number");
	if(readtimestep == timestep){
	  found=true;
	  break;
	}
      }
    }
    if(!found){
      string timestepindex = tname.str()+"/timestep.xml";      
      DOM_Text leader = indexDoc.createTextNode("\n\t");
      ts.appendChild(leader);
      DOM_Element newElem = indexDoc.createElement("timestep");
      ts.appendChild(newElem);
      ostringstream value;
      value << timestep;
      DOM_Text newVal = indexDoc.createTextNode(value.str().c_str());
      newElem.appendChild(newVal);
      newElem.setAttribute("href", timestepindex.c_str());
      DOM_Text trailer = indexDoc.createTextNode("\n");
      ts.appendChild(trailer);
    }
    
    ofstream indexOut(iname.c_str());
    indexOut << indexDoc << endl;     
  }
}

void DataArchiver::scheduleOutputTimestep(Dir& baseDir,
				    vector<DataArchiver::SaveItem>& saveLabels,
				    const LevelP& level, SchedulerP& sched)
{
  // Schedule a bunch o tasks - one for each variable, for each patch
  // This will need to change for parallel code
  int n=0;
  vector< SaveItem >::iterator saveIter;
  const PatchSet* patches = level->eachPatch();
  for(saveIter = saveLabels.begin(); saveIter!= saveLabels.end();
      saveIter++) {
    const MaterialSet* matls = (*saveIter).getMaterialSet();
    Task* t = scinew Task("DataArchiver::output", 
			  this, &DataArchiver::output,
			  &baseDir, (*saveIter).label_);
    t->requires(Task::NewDW, (*saveIter).label_, Ghost::None);
    sched->addTask(t, patches, matls);
    n++;
  }
   
  dbg << "Created " << n << " output tasks\n";
}

DOM_Document loadDocument(string xmlName)
{
   // Instantiate the DOM parser.
   DOMParser parser;
   parser.setDoValidation(false);
   
   SimpleErrorHandler handler;
   parser.setErrorHandler(&handler);
   
   parser.parse(xmlName.c_str());
   
   if(handler.foundError)
     throw InternalError("Error reading file: " + xmlName);
   
   return parser.getDocument();
}

const string
DataArchiver::getOutputLocation() const
{
    return d_dir.getName();
}

void DataArchiver::indexAddGlobals()
{
  // assume for now that global variables that get computed will not
  // change from timestep to timestep
  static bool wereGlobalsAdded = false;
   if (d_writeMeta && !wereGlobalsAdded) {
     wereGlobalsAdded = true;
      // add saved global (reduction) variables to index.xml
      string iname = d_dir.getName()+"/index.xml";
      DOM_Document indexDoc = loadDocument(iname);
      DOM_Node leader = indexDoc.createTextNode("\n");
      indexDoc.getDocumentElement().appendChild(leader);
      DOM_Node globals = indexDoc.createElement("globals");
      indexDoc.getDocumentElement().appendChild(globals);
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
	    DOM_Element newElem = indexDoc.createElement("variable");
	    DOM_Text leader = indexDoc.createTextNode("\n\t");
	    DOM_Text trailer = indexDoc.createTextNode("\n");
	    globals.appendChild(leader);
	    globals.appendChild(newElem);
	    globals.appendChild(trailer);
	    newElem.setAttribute("href", href.str().c_str());
	    newElem.setAttribute("type",
				 var->typeDescription()->getName().c_str());
	    newElem.setAttribute("name", var->getName().c_str());	 
	 }
      }
      
      ofstream indexOut(iname.c_str());
      indexOut << indexDoc << endl;
   }
}

void DataArchiver::outputReduction(const ProcessorGroup*,
				   const PatchSubset*,
				   const MaterialSubset* /*matls*/,
				   DataWarehouse* /*old_dw*/,
				   DataWarehouse* new_dw)
{
  // Dump the stuff in the reduction saveset into files in the uda
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
      out << setprecision(17) << d_currentTime << "\t";
      new_dw->print(out, var, matlIndex);
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
  dbg << "DataArchiver::outputCheckpointReduction called\n";
  // Dump the stuff in the reduction saveset into files in the uda

  for(int i=0;i<(int)d_checkpointReductionLabels.size();i++) {
    SaveItem& saveItem = d_checkpointReductionLabels[i];
    const VarLabel* var = saveItem.label_;
    const MaterialSubset* matls = saveItem.getMaterialSet()->getUnion();
    PatchSubset* patches = scinew PatchSubset(0);
    patches->add(0);
    output(world, patches, matls, old_dw, new_dw, &d_checkpointsDir, var);
    delete patches;
  }
}

void DataArchiver::output(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* /*old_dw*/,
			  DataWarehouse* new_dw,
			  Dir* p_dir,
			  const VarLabel* var)
{
  bool isReduction = var->typeDescription()->isReductionVariable();

  dbg << "output called ";
  if(patches->size() == 1 && !patches->get(1)){
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
  dbg << " at time: " << d_currentTimestep << "\n";
  
  ostringstream tname;
  tname << "t" << setw(4) << setfill('0') << d_currentTimestep;
  
  Dir tdir = p_dir->getSubdir(tname.str());
  
  string xmlFilename;
  string dataFilebase;
  string dataFilename;
  Mutex* pLock = 0;
  if (!isReduction) {
    ostringstream lname;
    lname << "l0"; // Hard coded - steve
    Dir ldir = tdir.getSubdir(lname.str());
    
    ostringstream pname;
    pname << "p" << setw(5) << setfill('0') << d_myworld->myrank();
    xmlFilename = ldir.getName() + "/" + pname.str() + ".xml";
    dataFilebase = pname.str() + ".data";
    dataFilename = ldir.getName() + "/" + dataFilebase;
    pLock = &d_outputLock;
  }
  else {
    xmlFilename =  tdir.getName() + "/global.xml";
    dataFilebase = "global.data";
    dataFilename = tdir.getName() + "/" + dataFilebase;
    pLock = &d_outputReductionLock;
  }

 pLock->lock();
  
  DOM_Document doc;
  ifstream test(xmlFilename.c_str());
  if(test){
    // Instantiate the DOM parser.
    DOMParser parser;
    parser.setDoValidation(false);
    
    SimpleErrorHandler handler;
    parser.setErrorHandler(&handler);
    
    // Parse the input file
    // No exceptions just yet, need to add
    
    parser.parse(xmlFilename.c_str());
    
    if(handler.foundError)
      throw InternalError("Error reading file: "+xmlFilename);
    
    // Add the parser contents to the ProblemSpecP d_doc
    doc = parser.getDocument();
  } else {
    DOM_DOMImplementation impl;
    
    doc = impl.createDocument(0,"Uintah_Output",
			      DOM_DocumentType());
  }

  // Find the end of the file
  DOM_Node n = doc.getDocumentElement();
  ASSERT(!n.isNull());
  n = findNode("Variable", n);
   
  long cur=0;
  while(!n.isNull()){
    DOM_Node endNode = findNode("end", n);
    ASSERT(!endNode.isNull());
    DOM_Node tn = findTextNode(endNode);
    DOMString val = tn.getNodeValue();
    char* s = val.transcode();
    long end = atoi(s);
    delete[] s;
    
    if(end > cur)
      cur=end;
    n=findNextNode("Variable", n);
  }


  DOM_Element rootElem = doc.getDocumentElement();
  // Open the data file
  int fd = open(dataFilename.c_str(), O_WRONLY|O_CREAT, 0666);
  if(fd == -1){
    cerr << "Cannot open dataFile: " << dataFilename << '\n';
    throw ErrnoException("DataArchiver::output (open call)", errno);
  }

  struct stat st;
  int s = fstat(fd, &st);
  if(s == -1){
    cerr << "Cannot fstat: " << dataFilename << '\n';
    throw ErrnoException("DataArchiver::output (stat call)", errno);
  }
  ASSERTEQ(cur, st.st_size);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int patchID = patch?patch->getID():-1;
    for(int m=0;m<matls->size();m++){
      int matlIndex = matls->get(m);
      DOM_Element pdElem = doc.createElement("Variable");
      rootElem.appendChild(pdElem);
      DOM_Text tailer = doc.createTextNode("\n");
      rootElem.appendChild(tailer);
  
      appendElement(pdElem, "variable", var->getName());
      appendElement(pdElem, "index", matlIndex);
      appendElement(pdElem, "patch", patchID);
      pdElem.setAttribute("type", var->typeDescription()->getName().c_str());

#ifdef __sgi
      off64_t ls = lseek64(fd, cur, SEEK_SET);
#else
      off_t ls = lseek(fd, cur, SEEK_SET);
#endif
      if(ls == -1)
	throw ErrnoException("DataArchiver::output (lseek64 call)", errno);

      // Pad appropriately
      if(cur%PADSIZE != 0){
	long pad = PADSIZE-cur%PADSIZE;
	char* zero = scinew char[pad];
	bzero(zero, pad);
	write(fd, zero, pad);
	cur+=pad;
	delete[] zero;
      }
      ASSERTEQ(cur%PADSIZE, 0);
      appendElement(pdElem, "start", cur);

      OutputContext oc(fd, cur, pdElem);
      new_dw->emit(oc, var, matlIndex, patch);
      appendElement(pdElem, "end", oc.cur);
      appendElement(pdElem, "filename", dataFilebase.c_str());
      s = fstat(fd, &st);
      if(s == -1)
	throw ErrnoException("DataArchiver::output (stat call)", errno);
      ASSERTEQ(oc.cur, st.st_size);
      cur=oc.cur;
    }
  }

  s = close(fd);
  if(s == -1)
    throw ErrnoException("DataArchiver::output (close call)", errno);

  ofstream out(xmlFilename.c_str());
  out << doc << endl;

  if(d_writeMeta){
    // Rewrite the index if necessary...
    string iname = p_dir->getName()+"/index.xml";
    DOM_Document indexDoc = loadDocument(iname);
    
    DOM_Node vs;
    string variableSection = (isReduction) ? "globals" : "variables";
	 
    vs = findNode(variableSection, indexDoc.getDocumentElement());
    if(vs == 0){
      DOM_Text leader = indexDoc.createTextNode("\n");
      indexDoc.getDocumentElement().appendChild(leader);
      vs = indexDoc.createElement(variableSection.c_str());
      indexDoc.getDocumentElement().appendChild(vs);
    }
    bool found=false;
    for(DOM_Node n = vs.getFirstChild(); n != 0; n=n.getNextSibling()){
      if(n.getNodeName().equals(DOMString("variable"))){
	DOM_NamedNodeMap attributes = n.getAttributes();
	DOM_Node varname = attributes.getNamedItem("name");
	if(varname == 0)
	  throw InternalError("varname not found");
	string vn = toString(varname.getNodeValue());
	if(vn == var->getName()){
	  found=true;
	  break;
	}
      }
    }
    if(!found){
      DOM_Text leader = indexDoc.createTextNode("\n\t");
      vs.appendChild(leader);
      DOM_Element newElem = indexDoc.createElement("variable");
      vs.appendChild(newElem);
      newElem.setAttribute("type", var->typeDescription()->getName().c_str());
      newElem.setAttribute("name", var->getName().c_str());
      DOM_Text trailer = indexDoc.createTextNode("\n");
      vs.appendChild(trailer);
    }
      
    ofstream indexOut(iname.c_str());
    indexOut << indexDoc << endl;
  }

 pLock->unlock();  
}

static Dir makeVersionedDir(const std::string nameBase)
{
   Dir dir;
   unsigned int dirMin = 0;
   unsigned int dirNum = 0;
   unsigned int dirMax = 0;

   bool dirCreated = false;
   while (!dirCreated) {
      ostringstream name;
      name << nameBase << "." << setw(3) << setfill('0') << dirNum;
      string dirName = name.str();
      
      try {
         dir = Dir::create(dirName);
            // throws an exception if dir exists

         dirMax = dirNum;
         if (dirMax == dirMin)
            dirCreated = true;
         else
            dir.remove();
      } catch (ErrnoException& e) {
         if (e.getErrno() != EEXIST)
            throw;

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
   int rc = lstat(nameBase.c_str(), &sb);
   if ((rc != 0) && (errno == ENOENT))
      make_link = true;
   else if ((rc == 0) && (S_ISLNK(sb.st_mode))) {
      unlink(nameBase.c_str());
      make_link = true;
   }
   if (make_link)
      symlink(dir.getName().c_str(), nameBase.c_str());
   
   return Dir(dir.getName());
}

void  DataArchiver::initSaveLabels(SchedulerP& sched)
{
  dbg << "initSaveLabels called\n";
   SaveItem saveItem;
   d_saveReductionLabels.clear();
   d_saveLabels.clear();
  
   d_saveLabels.reserve(d_saveLabelNames.size());
   Scheduler::VarLabelMaterialMap* pLabelMatlMap;
   pLabelMatlMap = sched->makeVarLabelMaterialMap();
   for (list<SaveNameItem>::iterator it = d_saveLabelNames.begin();
        it != d_saveLabelNames.end(); it++) {
     
      VarLabel* var = VarLabel::find((*it).labelName);
      if (var == NULL)
         throw ProblemSetupException((*it).labelName +
				     " variable not found to save.");

      if ((*it).compressionMode != "")
	var->setCompressionMode((*it).compressionMode);
      
      Scheduler::VarLabelMaterialMap::iterator found =
	pLabelMatlMap->find(var->getName());

      if (found == pLabelMatlMap->end())
         throw ProblemSetupException((*it).labelName +
				  " variable not computed for saving.");
      
      saveItem.label_ = var;
      ConsecutiveRangeSet matlsToSave =
	(ConsecutiveRangeSet((*found).second)).intersected((*it).matls);
      saveItem.setMaterials(matlsToSave, prevMatls_, prevMatlSet_);
      
      if (((*it).matls != ConsecutiveRangeSet::all) &&
	  ((*it).matls != matlsToSave)) {
         throw ProblemSetupException((*it).labelName +
	  " variable not computed for all materials specified to save.");
      }
      
      if (saveItem.label_->typeDescription()->isReductionVariable())
         d_saveReductionLabels.push_back(saveItem);
      else
         d_saveLabels.push_back(saveItem);
   }
   d_saveLabelNames.clear();
   delete pLabelMatlMap;
}


void DataArchiver::initCheckpoints(SchedulerP& sched)
{
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
   for (mapIter = label_matl_map.begin();
        mapIter != label_matl_map.end(); mapIter++) {
      VarLabel* var = VarLabel::find((*mapIter).first);
      if (var == NULL)
         throw ProblemSetupException((*mapIter).first +
				  " variable not found to checkpoint.");

      saveItem.label_ = var;
      saveItem.setMaterials((*mapIter).second, prevMatls_, prevMatlSet_);

      if (saveItem.label_->typeDescription()->isReductionVariable())
         d_checkpointReductionLabels.push_back(saveItem);
      else
         d_checkpointLabels.push_back(saveItem);
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
	 iter != matls.end(); iter++)
      matlVec.push_back(*iter);
    matlSet_->addAll(matlVec);
    prevMatlSet = matlSet_;
    prevMatls = matls;
  }
}

bool DataArchiver::need_recompile(double time, double dt ,
				  const LevelP& level )
{
  d_currentTime=time+dt;
  dbg << "DataArchiver::need_recompile called\n";
  d_currentTimestep++;
  bool recompile=false;
  bool do_output=false;
  if (d_outputInterval != 0 && time+dt >= d_nextOutputTime){
    do_output=true;
    if(!d_wasOutputTimestep)
      recompile=true;
  } else {
    if(d_wasOutputTimestep)
      recompile=true;
  }
  if (d_checkpointInterval != 0 && time+dt >= d_nextCheckpointTime) {
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
    beginOutputTimestep(time, dt, level);
  else
    d_wasCheckpointTimestep=d_wasOutputTimestep=false;
  if(recompile)
    dbg << "We do request recompile\n";
  else
    dbg << "We do not request recompile\n";
  return recompile;
}
