
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
#include <PSECore/XMLUtil/SimpleErrorHandler.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <iomanip>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <strings.h>
#include <unistd.h>

#define PADSIZE 1024L

using namespace Uintah;
using namespace std;
using namespace SCICore::OS;
using namespace SCICore::Exceptions;
using namespace PSECore::XMLUtil;

static Dir makeVersionedDir(const std::string nameBase);

DataArchiver::DataArchiver(int MpiRank, int MpiProcesses)
   : UintahParallelComponent(MpiRank, MpiProcesses)
{
}

DataArchiver::~DataArchiver()
{
}

void DataArchiver::problemSetup(const ProblemSpecP& params)
{
   ProblemSpecP p = params->findBlock("DataArchiver");
   p->require("filebase", d_filebase);
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1;

   d_currentTimestep = 0;

   if (d_outputInterval == 0) 
	return;

   d_dir = makeVersionedDir(d_filebase);

   DOM_DOMImplementation impl;
    
   DOM_Document doc = impl.createDocument(0,"Uintah_DataArchive",
					  DOM_DocumentType());
   DOM_Element rootElem = doc.getDocumentElement();

   appendElement(rootElem, "numberOfProcessors", d_MpiProcesses);

   DOM_Element metaElem = doc.createElement("Meta");
   rootElem.appendChild(metaElem);
   appendElement(metaElem, "username", getenv("LOGNAME"));
   time_t t = time(NULL) ;
   appendElement(metaElem, "date", ctime(&t));

   string iname = d_filebase+"/index.xml";
   ofstream out(iname.c_str());
   out << doc << endl;

   string inputname = d_filebase+"/input.xml";
   ofstream out2(inputname.c_str());
   out2 << params->getNode().getOwnerDocument() << endl;
}

void DataArchiver::finalizeTimestep(double time, double delt,
				    const LevelP& level, SchedulerP& sched,
				    DataWarehouseP& /*old_dw*/,
				    DataWarehouseP& new_dw)
{
  
   if (d_outputInterval == 0)
      return;
 
   int timestep = d_currentTimestep;

   // Schedule task to dump out integrated data at every timestep
   vector<const VarLabel*> ivars;
   new_dw->getIntegratedSaveSet(ivars);


   Task* t = scinew Task("DataArchiver::outputReduction", new_dw, new_dw,
		  this, &DataArchiver::outputReduction);

    for(int i=0;i<ivars.size();i++){
      t->requires(new_dw, ivars[i]) ;
    }

   sched->addTask(t);

   cerr << "Created reduction variable output task" << endl;

   if((d_currentTimestep++ % d_outputInterval) != 0)
      return;

   ostringstream tname;
   tname << "t" << setw(4) << setfill('0') << timestep;

   // Create the directory for this timestep, if necessary
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
	 Level::const_patchIterator iter;
	 for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
	    const Patch* patch=*iter;
	    DOM_Element patchElem = doc.createElement("Patch");
	    levelElem.appendChild(patchElem);
	    appendElement(patchElem, "id", patch->getID());
	    appendElement(patchElem, "lowIndex", patch->getCellLowIndex());
	    appendElement(patchElem, "highIndex", patch->getCellHighIndex());
	    appendElement(patchElem, "resolution", patch->getNCells());
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
      for(int i=0;i<d_MpiProcesses;i++){
	 ostringstream pname;
	 pname << lname.str() << "/p" << setw(5) << setfill('0') << i << ".xml";
	 DOM_Text leader = doc.createTextNode("\n\t");
	 dataElem.appendChild(leader);
	 DOM_Element df = doc.createElement("Datafile");
	 dataElem.appendChild(df);
	 df.setAttribute("href", pname.str().c_str());
	 ostringstream labeltext;
	 labeltext << "Processor " << i << " of " << d_MpiProcesses;
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

   
   vector<const VarLabel*> vars;
   vector<int> number;
   new_dw->getSaveSet(vars, number);

   // Schedule a bunch o tasks - one for each variable, for each patch
   // This will need to change for parallel code
   int n=0;
   Level::const_patchIterator iter;
   for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;
      for(int i=0;i<vars.size();i++){
	 for(int j=0;j<number[i];j++){
	    Task* t = scinew Task("DataArchiver::output", patch, new_dw, new_dw,
				  this, &DataArchiver::output, timestep,
				  vars[i], j);
	    t->requires(new_dw, vars[i], j, patch, Ghost::None);
	    sched->addTask(t);
	    n++;
	 }
      }
   }
   cerr << "Created " << n << " output tasks\n";
}

static bool get(const DOM_Node& node, int &value)
{
   for (DOM_Node child = node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 DOMString val = child.getNodeValue();
	 char* s = val.transcode();
	 value = atoi(s);
	 delete[] s;
	 return true;
      }
   }
   return false;
}

void DataArchiver::outputReduction(const ProcessorContext*,
//			  const Patch* patch,
			  DataWarehouseP& /*old_dw*/,
			  DataWarehouseP& new_dw)
{
   // Dump the stuff in the reduction saveset into an
   // ofstream

   vector<const VarLabel*> ivars;
   new_dw->getIntegratedSaveSet(ivars);

   static ofstream intout("intquan.dat");
   new_dw->emit(intout, ivars);

}

void DataArchiver::output(const ProcessorContext*,
			  const Patch* patch,
			  DataWarehouseP& /*old_dw*/,
			  DataWarehouseP& new_dw,
			  int timestep,
			  const VarLabel* var,
			  int matlIndex)
{
   if (d_outputInterval == 0)
      return;

   cerr << "output called on patch: " << patch->getID() << ", variable: " << var->getName() << ", material: " << matlIndex << " at time: " << timestep << "\n";
   
   ostringstream tname;
   tname << "t" << setw(4) << setfill('0') << timestep;

   Dir tdir = d_dir.getSubdir(tname.str());

   ostringstream lname;
   lname << "l0"; // Hard coded - steve
   Dir ldir = tdir.getSubdir(lname.str());


   ostringstream pname;
   pname << ldir.getName() << "/p" << setw(5) << setfill('0') << d_MpiRank << ".xml";

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
   base << "p" << setw(5) << setfill('0') << d_MpiRank << ".data";
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
   off64_t ls = lseek64(fd, cur, SEEK_SET);
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

   // Rewrite the index if necessary...
   string iname = d_filebase+"/index.xml";
   // Instantiate the DOM parser.
   DOMParser parser;
   parser.setDoValidation(false);

   SimpleErrorHandler handler;
   parser.setErrorHandler(&handler);

   // Parse the input file
   // No exceptions just yet, need to add

   parser.parse(iname.c_str());

   if(handler.foundError)
      throw InternalError("Error reading file: "+pname_s);

   // Add the parser contents to the ProblemSpecP d_doc
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

static Dir makeVersionedDir(const std::string nameBase)
{
   Dir dir;
   string dirName = nameBase;
   unsigned int dirMin = 0;
   unsigned int dirNum = 0;
   unsigned int dirMax = 0;

   bool dirCreated = false;
   while (!dirCreated) {
	   try {
         dir = Dir::create(dirName);

         dirMax = dirNum;
         if (dirMax == dirMin)
            dirCreated = true;
         else
            dir.remove();

      } catch (ErrnoException& e) {
         if (e.getErrno() != EEXIST)
            throw e;

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
	
         ostringstream name;
         name << nameBase << "." << setw(3) << setfill('0') << dirNum;
         dirName = name.str();
      }
   }

   return Dir(dir.getName());
}

//
// $Log$
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
