#include <Uintah/Components/DataArchiver/DataArchiver.h>
#include <SCICore/Exceptions/ErrnoException.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Util/FancyAssert.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/OutputContext.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Parallel/Parallel.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <Uintah/Exceptions/ProblemSetupException.h>
#include <PSECore/XMLUtil/SimpleErrorHandler.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <SCICore/Util/DebugStream.h>
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
#ifdef __aix
#include <time.h>
#endif

#define PADSIZE 1024L

using namespace Uintah;
using namespace std;
using namespace SCICore::OS;
using namespace SCICore::Exceptions;
using namespace PSECore::XMLUtil;
using namespace SCICore::Util;

static Dir makeVersionedDir(const std::string nameBase);
static DebugStream dbg("DataArchiver", false);

DataArchiver::DataArchiver(const ProcessorGroup* myworld)
   : UintahParallelComponent(myworld)
{
  d_wasOutputTimestep = false;
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

   SaveNameItem saveItem;
   ProblemSpecP save = p->findBlock("save");
   while (save != NULL) {
      map<string, string> attributes;
      save->getAttributes(attributes);
      saveItem.labelName = attributes["label"];
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
   
   d_currentTimestep = 0;
   d_lastTimestepLocation = "invalid";
   d_wasOutputTimestep = false;

   if (d_outputInterval == 0.0) 
	return;

   d_nextOutputTime=0.0;

   if(Parallel::usingMPI()){
      // See if we have a shared filesystem
      bool shared;
      if(d_myworld->myrank() == 0){
	 d_dir = makeVersionedDir(d_filebase);
	 string tmpname = d_dir.getName()+"/tmp.tmp";
	 ostringstream test_string;
	 char hostname[MAXHOSTNAMELEN];
	 if(gethostname(hostname, MAXHOSTNAMELEN) != 0)
	    strcpy(hostname, "unknown???");
	 test_string << hostname << " " << getpid() << '\n';
	 {
	    ofstream tmpout(tmpname.c_str());
	    tmpout << test_string.str();
	 }
	 int outlen = (int)(test_string.str().length()+tmpname.length()+2);
	 char* outbuf = new char[outlen];
	 strcpy(outbuf, test_string.str().c_str());
	 strcpy(outbuf+test_string.str().length()+1, tmpname.c_str());
	 MPI_Bcast(&outlen, 1, MPI_INT, 0, d_myworld->getComm());
	 MPI_Bcast(outbuf, outlen, MPI_CHAR, 0, d_myworld->getComm());
	 delete[] outbuf;
	 shared=true;
      } else {
	 int inlen;
	 MPI_Bcast(&inlen, 1, MPI_INT, 0, d_myworld->getComm());
	 char* inbuf = new char[inlen];
	 MPI_Bcast(inbuf, inlen, MPI_CHAR, 0, d_myworld->getComm());
	 char* test_string = inbuf;
	 char* tmpname = test_string+strlen(test_string)+1;
	 ifstream tmpin(tmpname);
	 if(tmpin){
	    char* in = new char[strlen(test_string)+1];
	    tmpin.read(in, (int)strlen(test_string));
	    in[strlen(test_string)]=0;
	    if(strcmp(test_string, in) != 0){
	       cerr << "Different strings?\n";
	       shared=false;
	    } else {
	       shared=true;
	    }
	 } else {
	    shared=false;
	 }
	 char* p = tmpname+strlen(tmpname)-1;
	 while(p>tmpname && *p != '/')
	    p--;
	 *p=0;
	 d_dir = Dir(tmpname);
#if 0
	 cerr << d_myworld->myrank() << " dir=" << d_dir.getName();
	 if(shared)
	    cerr << " shared\n";
	 else
	    cerr << " NOT shared\n";
#endif
	 delete[] inbuf;
      }
      int s = shared;
      int allshared;
      MPI_Allreduce(&s, &allshared, 1, MPI_INT, MPI_MIN, d_myworld->getComm());
      if(allshared){
	 if(d_myworld->myrank() == 0){
	    d_writeMeta = true;
	 } else {
	    d_writeMeta = false;
	 }
      } else {
	 d_writeMeta = true;
      }
      if(d_myworld->myrank() == 0){
	 string tmpname = d_dir.getName()+"/tmp.tmp";
	 unlink(tmpname.c_str());
      }
   } else {
      d_dir = makeVersionedDir(d_filebase);
      d_writeMeta = true;
   }

   if(d_writeMeta){
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

      string iname = d_dir.getName()+"/index.xml";
      ofstream out(iname.c_str());
      out << doc << endl;

      string inputname = d_dir.getName()+"/input.xml";
      ofstream out2(inputname.c_str());
      out2 << params->getNode().getOwnerDocument() << endl;
   }
}

void DataArchiver::finalizeTimestep(double time, double delt,
				    const LevelP& level, SchedulerP& sched,
				    DataWarehouseP& new_dw)
{
   if (d_saveLabelNames.size() > 0 &&
       !(time == 0 && delt == 0) /* skip the initialization timestep for this
				    because it needs all computes to be set
				    to find the save labels */)
      initSaveLabels(sched);
  
   if (d_outputInterval == 0.0)
      return;
 
   int timestep = d_currentTimestep;

   // Schedule task to dump out reduction variables at every timestep
   Task* t = scinew Task("DataArchiver::outputReduction", new_dw, new_dw,
			 this, &DataArchiver::outputReduction, time);

   for(int i=0;i<(int)d_saveReductionLabels.size();i++) {
      SaveItem& saveItem = d_saveReductionLabels[i];
      const VarLabel* var = saveItem.label;
      for (ConsecutiveRangeSet::iterator matIt = saveItem.matls.begin();
	   matIt != saveItem.matls.end(); matIt++) {     
	 t->requires(new_dw, var, *matIt) ;
      }
   }

   sched->addTask(t);

   dbg << "Created reduction variable output task" << endl;

   d_currentTimestep++;
   if(time<d_nextOutputTime) {
      d_wasOutputTimestep = false;
      return;
   }
   d_wasOutputTimestep = true;
   
//   if((d_currentTimestep++ % d_outputInterval) != 0)
//      return;

   d_nextOutputTime+=d_outputInterval;

   ostringstream tname;
   tname << "t" << setw(4) << setfill('0') << timestep;
   d_lastTimestepLocation = d_dir.getName() + "/" + tname.str();

   // Create the directory for this timestep, if necessary
   if(d_writeMeta){
      Dir tdir;
      try {
	 tdir = d_dir.createSubdir(tname.str());

	 DOM_DOMImplementation impl;
    
	 DOM_Document doc = impl.createDocument(0,"Uintah_timestep",
						DOM_DocumentType());
	 DOM_Element rootElem = doc.getDocumentElement();

	 DOM_Element timeElem = doc.createElement("Time");
	 rootElem.appendChild(timeElem);

	 appendElement(timeElem, "timestepNumber", timestep);
	 appendElement(timeElem, "currentTime", time);
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
	 lname << "l0"; // Hard coded - steve
	 for(int i=0;i<d_myworld->size();i++){
	    ostringstream pname;
	    pname << lname.str() << "/p" << setw(5) << setfill('0') << i << ".xml";
	    DOM_Text leader = doc.createTextNode("\n\t");
	    dataElem.appendChild(leader);
	    DOM_Element df = doc.createElement("Datafile");
	    dataElem.appendChild(df);
	    df.setAttribute("href", pname.str().c_str());
	    ostringstream labeltext;
	    labeltext << "Processor " << i << " of " << d_myworld->size();
	    DOM_Text label = doc.createTextNode(labeltext.str().c_str());
	    df.appendChild(label);
	    DOM_Text trailer = doc.createTextNode("\n");
	    dataElem.appendChild(trailer);
	 }
	 string name = tdir.getName()+"/timestep.xml";
	 ofstream out(name.c_str());
	 out << doc << endl;
	 
      } catch(ErrnoException& e) {
	 if(e.getErrno() != EEXIST)
	    throw;
	 tdir = d_dir.getSubdir(tname.str());
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
   }
      
   // Schedule a bunch o tasks - one for each variable, for each patch
   // This will need to change for parallel code
   int n=0;
   Level::const_patchIterator iter;
   for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;
      vector< SaveItem >::iterator saveIter;
      for(saveIter = d_saveLabels.begin(); saveIter!= d_saveLabels.end();
	  saveIter++) {
	 ConsecutiveRangeSet::iterator matlIter = (*saveIter).matls.begin();
	 for ( ; matlIter != (*saveIter).matls.end(); matlIter++) {
	    Task* t = scinew Task("DataArchiver::output", patch, new_dw,
				  new_dw, this, &DataArchiver::output,
				  timestep, (*saveIter).label, *matlIter);
	    t->requires(new_dw, (*saveIter).label, *matlIter, patch,
			Ghost::None);
	    sched->addTask(t);
	    n++;
	 }
      }
   }
   dbg << "Created " << n << " output tasks\n";
}

const string
DataArchiver::getOutputLocation() const
{
    return d_dir.getName();
}

void DataArchiver::outputReduction(const ProcessorGroup*,
				   DataWarehouseP& /*old_dw*/,
				   DataWarehouseP& new_dw,
				   double time)
{
   // Dump the stuff in the reduction saveset into files in the uda

   for(int i=0;i<(int)d_saveReductionLabels.size();i++) {
      SaveItem& saveItem = d_saveReductionLabels[i];
      const VarLabel* var = saveItem.label;
      for (ConsecutiveRangeSet::iterator matIt = saveItem.matls.begin();
	   matIt != saveItem.matls.end(); matIt++) {
         int matlIndex = *matIt;
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
	 out << setprecision(17) << time << "\t";
	 new_dw->emit(out, var, matlIndex);
	 out << "\n";
      }
   }
}

void DataArchiver::output(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP& /*old_dw*/,
			  DataWarehouseP& new_dw,
			  int timestep,
			  const VarLabel* var,
			  int matlIndex)
{
   if (d_outputInterval == 0.0)
      return;

   dbg << "output called on patch: " << patch->getID() << ", variable: " << var->getName() << ", material: " << matlIndex << " at time: " << timestep << "\n";
   
   ostringstream tname;
   tname << "t" << setw(4) << setfill('0') << timestep;

   Dir tdir = d_dir.getSubdir(tname.str());

   ostringstream lname;
   lname << "l0"; // Hard coded - steve
   Dir ldir = tdir.getSubdir(lname.str());


   ostringstream pname;
   pname << ldir.getName() << "/p" << setw(5) << setfill('0') << d_myworld->myrank() << ".xml";

   DOM_Document doc;
   string pname_s = pname.str();
   ifstream test(pname_s.c_str());
   if(test){
      // Instantiate the DOM parser.
      DOMParser parser;
      parser.setDoValidation(false);

      SimpleErrorHandler handler;
      parser.setErrorHandler(&handler);

      // Parse the input file
      // No exceptions just yet, need to add

      parser.parse(pname_s.c_str());

      if(handler.foundError)
	 throw InternalError("Error reading file: "+pname_s);

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
   DOM_Element pdElem = doc.createElement("Variable");
   rootElem.appendChild(pdElem);
   DOM_Text tailer = doc.createTextNode("\n");
   rootElem.appendChild(tailer);

   appendElement(pdElem, "variable", var->getName());
   appendElement(pdElem, "index", matlIndex);
   appendElement(pdElem, "patch", patch->getID());
   pdElem.setAttribute("type", var->typeDescription()->getName().c_str());

   // Open the data file
   ostringstream base;
   base << "p" << setw(5) << setfill('0') << d_myworld->myrank() << ".data";
   ostringstream dname;
   dname << ldir.getName() << "/" << base.str();
   string datafile = dname.str();
   int fd = open(datafile.c_str(), O_WRONLY|O_CREAT, 0666);
   if(fd == -1)
      throw ErrnoException("DataArchiver::output (open call)", errno);

   struct stat st;
   int s = fstat(fd, &st);
   if(s == -1)
      throw ErrnoException("DataArchiver::output (stat call)", errno);
   ASSERTEQ(cur, st.st_size);
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
   appendElement(pdElem, "filename", base.str());
   s = fstat(fd, &st);
   if(s == -1)
      throw ErrnoException("DataArchiver::output (stat call)", errno);
   ASSERTEQ(oc.cur, st.st_size);

   s = close(fd);
   if(s == -1)
      throw ErrnoException("DataArchiver::output (close call)", errno);

   ofstream out(pname_s.c_str());
   out << doc << endl;

   if(d_writeMeta){
      // Rewrite the index if necessary...
      string iname = d_dir.getName()+"/index.xml";
      // Instantiate the DOM parser.
      DOMParser parser;
      parser.setDoValidation(false);

      SimpleErrorHandler handler;
      parser.setErrorHandler(&handler);

      parser.parse(iname.c_str());

      if(handler.foundError)
	 throw InternalError("Error reading file: "+pname_s);

      DOM_Document topDoc = parser.getDocument();
      DOM_Node ts = findNode("timesteps", topDoc.getDocumentElement());
      if(ts == 0){
	 ts = topDoc.createElement("timesteps");
	 topDoc.getDocumentElement().appendChild(ts);
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
	 DOM_Text leader = topDoc.createTextNode("\n\t");
	 ts.appendChild(leader);
	 DOM_Element newElem = topDoc.createElement("timestep");
	 ts.appendChild(newElem);
	 ostringstream value;
	 value << timestep;
	 DOM_Text newVal = topDoc.createTextNode(value.str().c_str());
	 newElem.appendChild(newVal);
	 newElem.setAttribute("href", timestepindex.c_str());
	 DOM_Text trailer = topDoc.createTextNode("\n");
	 ts.appendChild(trailer);
      }

      DOM_Node vs = findNode("variables", topDoc.getDocumentElement());
      if(vs == 0){
	 vs = topDoc.createElement("variables");
	 topDoc.getDocumentElement().appendChild(vs);
      }
      found=false;
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
	 DOM_Text leader = topDoc.createTextNode("\n\t");
	 vs.appendChild(leader);
	 DOM_Element newElem = topDoc.createElement("variable");
	 vs.appendChild(newElem);
	 newElem.setAttribute("type", var->typeDescription()->getName().c_str());
	 newElem.setAttribute("name", var->getName().c_str());
	 DOM_Text trailer = topDoc.createTextNode("\n");
	 vs.appendChild(trailer);
      }
      
      ofstream topout(iname.c_str());
      topout << topDoc << endl;
   }
}

static Dir makeVersionedDir(const std::string nameBase)
{
   Dir dir;
   string dirName;
   unsigned int dirMin = 0;
   unsigned int dirNum = 0;
   unsigned int dirMax = 0;

   bool dirCreated = false;
   while (!dirCreated) {
      ostringstream name;
      name << nameBase << "." << setw(3) << setfill('0') << dirNum;
      dirName = name.str();
      
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
   SaveItem saveItem;
  
   d_saveLabels.resize(d_saveLabelNames.size());
   Scheduler::VarLabelMaterialMap* pLabelMatlMap;
   pLabelMatlMap = sched->makeVarLabelMaterialMap();
   for (list<SaveNameItem>::iterator it = d_saveLabelNames.begin();
        it != d_saveLabelNames.end(); it++) {
      Scheduler::VarLabelMaterialMap::iterator found =
	 pLabelMatlMap->find((*it).labelName);

      if (found == pLabelMatlMap->end())
         throw ProblemSetupException((*it).labelName +
				     " variable label not found to save.");
      
      saveItem.label = (*found).second.first;
      saveItem.matls = ConsecutiveRangeSet((*found).second.second);
      saveItem.matls = saveItem.matls.intersected((*it).matls);
      
      if (saveItem.label->typeDescription()->isReductionVariable())
         d_saveReductionLabels.push_back(saveItem);
      else
         d_saveLabels.push_back(saveItem);
   }
   d_saveLabelNames.clear();
}

//
// $Log$
// Revision 1.23  2000/12/07 01:27:29  witzel
// Added some changes I forgot to make pertaining saving reduction variables
// (creating outputReduction task).
//
// Revision 1.22  2000/12/06 23:59:40  witzel
// Added variable save functionality via the DataArchiver problem spec
//
// Revision 1.21  2000/09/29 05:41:57  sparker
// Quiet g++ warnings
//
// Revision 1.20  2000/09/28 17:54:29  bigler
// Added #include <time.h> for aix
//
// Revision 1.19  2000/09/25 18:06:55  sparker
// linux/g++ changes
//
// Revision 1.18  2000/09/08 17:00:10  witzel
// Added functions for getting the last timestep directory, the current
// timestep, and whether the last timestep was one in which data was
// output.  These functions are needed by the Scheduler to archive taskgraph
// data.
//
// Revision 1.17  2000/08/25 18:04:42  sparker
// Yanked print statement
//
// Revision 1.16  2000/08/25 17:41:15  sparker
// All output from an MPI run now goes into a single UDA dir
//
// Revision 1.15  2000/07/26 20:14:09  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.14  2000/06/27 17:08:32  bigler
// Steve moved some functions around for me.
//
// Revision 1.13  2000/06/17 07:06:30  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.12  2000/06/16 19:48:19  sparker
// Use updated output interface
//
// Revision 1.11  2000/06/15 21:56:59  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.10  2000/06/14 21:50:54  guilkey
// Changed DataArchiver to create uda files based on a per time basis
// rather than a per number of timesteps basis.
//
// Revision 1.9  2000/06/05 20:46:32  jehall
// - Removed special case for first output dir -- it is now <base>.000
// - Added symlink generation from <base> to latest output dir, e.g.
//   disks.uda -> disks.uda.002
//
// Revision 1.8  2000/06/05 19:51:26  guilkey
// Removed gratuitous screen output.
//
// Revision 1.7  2000/06/03 05:24:25  sparker
// Finished/changed reduced variable emits
// Fixed bug in directory version numbers where the index was getting
//   written to the base file directory instead of the versioned one.
//
// Revision 1.6  2000/06/02 20:25:59  jas
// If outputInterval is 0 (input file), no output is generated.
//
// Revision 1.5  2000/06/01 23:09:38  guilkey
// Added beginnings of code to store integrated quantities.
//
// Revision 1.4  2000/05/31 15:21:50  jehall
// - Added output dir versioning
//
// Revision 1.3  2000/05/30 20:18:54  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.2  2000/05/20 08:09:03  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.1  2000/05/15 19:39:35  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
//
