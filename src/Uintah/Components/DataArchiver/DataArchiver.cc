
#include <Uintah/Components/DataArchiver/DataArchiver.h>
#include <SCICore/Exceptions/ErrnoException.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Util/FancyAssert.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/OutputContext.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/Scheduler.h>
#include <iomanip>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <sax/ErrorHandler.hpp>
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
void          outputContent(ostream& target, const DOMString &s);
ostream& operator<<(ostream& target, const DOMString& toWrite);
ostream& operator<<(ostream& target, DOM_Node& toWrite);

class DAErrorHandler : public ErrorHandler {
public:
    bool foundError;

    DAErrorHandler();
    ~DAErrorHandler();

    void warning(const SAXParseException& e);
    void error(const SAXParseException& e);
    void fatalError(const SAXParseException& e);
    void resetErrors();

private :
    DAErrorHandler(const DAErrorHandler&);
    void operator=(const DAErrorHandler&);
};

DAErrorHandler::DAErrorHandler()
{
    foundError=false;
}

DAErrorHandler::~DAErrorHandler()
{
}

static void postMessage(const string& errmsg, bool err=true)
{
   if(err)
      cerr << "ERROR: ";
   else
      cerr << "WARNING: ";
   cerr << errmsg << '\n';
}

static string xmlto_string(const XMLCh* const str)
{
    char* s = XMLString::transcode(str);
    string ret = string(s);
    delete[] s;
    return ret;
}

static string to_string(int i)
{
    char buf[20];
    sprintf(buf, "%d", i);
    return string(buf);
}

void DAErrorHandler::error(const SAXParseException& e)
{
    foundError=true;
    postMessage(string("Error at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void DAErrorHandler::fatalError(const SAXParseException& e)
{
    foundError=true;
    postMessage(string("Fatal Error at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void DAErrorHandler::warning(const SAXParseException& e)
{
    postMessage(string("Warning at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void DAErrorHandler::resetErrors()
{
}

// ---------------------------------------------------------------------------
//
//  ostream << DOM_Node   
//
//                Stream out a DOM node, and, recursively, all of its children.
//                This function is the heart of writing a DOM tree out as
//                XML source.  Give it a document node and it will do the whole thing.
//
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, const DOM_Node& toWrite)
{
   // Get the name and value out for convenience
   DOMString   nodeName = toWrite.getNodeName();
   DOMString   nodeValue = toWrite.getNodeValue();
   
   switch (toWrite.getNodeType()) {
   case DOM_Node::TEXT_NODE:
      {
	 outputContent(target, nodeValue);
	 break;
      }
   
   case DOM_Node::PROCESSING_INSTRUCTION_NODE :
      {
	 target  << "<?"
		 << nodeName
		 << ' '
		 << nodeValue
		 << "?>";
	 break;
      }
   
   case DOM_Node::DOCUMENT_NODE :
      {
	 // Bug here:  we need to find a way to get the encoding name
	 //   for the default code page on the system where the
	 //   program is running, and plug that in for the encoding
	 //   name.  
	 target << "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
	 DOM_Node child = toWrite.getFirstChild();
	 while( child != 0)
            {
	       target << child << endl;
	       child = child.getNextSibling();
            }
	 
	 break;
      }
   
   case DOM_Node::ELEMENT_NODE :
      {
	 // Output the element start tag.
	 target << '<' << nodeName;
	 
	 // Output any attributes on this element
	 DOM_NamedNodeMap attributes = toWrite.getAttributes();
	 int attrCount = attributes.getLength();
	 for (int i = 0; i < attrCount; i++) {
	    DOM_Node  attribute = attributes.item(i);
	    
	    target  << ' ' << attribute.getNodeName()
		    << " = \"";
	    //  Note that "<" must be escaped in attribute values.
	    outputContent(target, attribute.getNodeValue());
	    target << '"';
	 }
	 
	 //
	 //  Test for the presence of children, which includes both
	 //  text content and nested elements.
	 //
	 DOM_Node child = toWrite.getFirstChild();
	 if (child != 0) {
	    // There are children. Close start-tag, and output children.
	    target << ">";
	    while( child != 0) {
	       target << child;
	       child = child.getNextSibling();
	    }

	    // Done with children.  Output the end tag.
	    target << "</" << nodeName << ">";
	 } else {
	    //
	    //  There were no children.  Output the short form close of the
	    //  element start tag, making it an empty-element tag.
	    //
	    target << "/>";
	 }
	 break;
      }
   
   case DOM_Node::ENTITY_REFERENCE_NODE:
      {
	 DOM_Node child;
	 for (child = toWrite.getFirstChild(); child != 0; child = child.getNextSibling())
	    target << child;
	 break;
      }
   
   case DOM_Node::CDATA_SECTION_NODE:
      {
	 target << "<![CDATA[" << nodeValue << "]]>";
	 break;
      }
   
   case DOM_Node::COMMENT_NODE:
      {
	 target << "<!--" << nodeValue << "-->";
	 break;
      }
   
   default:
      cerr << "Unrecognized node type = "
	   << (long)toWrite.getNodeType() << endl;
   }
   return target;
}


// ---------------------------------------------------------------------------
//
//  outputContent  - Write document content from a DOMString to a C++ ostream.
//                   Escape the XML special characters (<, &, etc.) unless this
//                   is suppressed by the command line option.
//
// ---------------------------------------------------------------------------
void outputContent(ostream& target, const DOMString &toWrite)
{
   int            length = toWrite.length();
   const XMLCh*   chars  = toWrite.rawBuffer();
   
   int index;
   for (index = 0; index < length; index++) {
      switch (chars[index]) {
      case chAmpersand :
	 target << "&amp;";
	 break;
	 
      case chOpenAngle :
	 target << "&lt;";
	 break;
	 
      case chCloseAngle:
	 target << "&gt;";
	 break;
	 
      case chDoubleQuote :
	 target << "&quot;";
	 break;
	 
      default:
	 // If it is none of the special characters, print it as such
	 target << toWrite.substringData(index, 1);
	 break;
      }
   }
}


// ---------------------------------------------------------------------------
//
//  ostream << DOMString    Stream out a DOM string.
//                          Doing this requires that we first transcode
//                          to char * form in the default code page
//                          for the system
//
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, const DOMString& s)
{
   char *p = s.transcode();
   target << p;
   delete [] p;
   return target;
}

DataArchiver::DataArchiver(int MpiRank, int MpiProcesses)
   : UintahParallelComponent(MpiRank, MpiProcesses)
{
}

DataArchiver::~DataArchiver()
{
}

void appendElement(DOM_Element& root, const DOMString& name,
		   const std::string& value)
{
   DOM_Text leader = root.getOwnerDocument().createTextNode("\n\t");
   root.appendChild(leader);
   DOM_Element newElem = root.getOwnerDocument().createElement(name);
   root.appendChild(newElem);
   DOM_Text newVal = root.getOwnerDocument().createTextNode(value.c_str());
   newElem.appendChild(newVal);
   DOM_Text trailer = root.getOwnerDocument().createTextNode("\n");
   root.appendChild(trailer);
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   int value)
{
   ostringstream val;
   val << value;
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   const IntVector& value)
{
   ostringstream val;
   val << '[' << value.x() << ", " << value.y() << ", " << value.z() << ']';
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   const Point& value)
{
   ostringstream val;
   val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   long value)
{
   ostringstream val;
   val << value;
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   double value)
{
   ostringstream val;
   val << setprecision(17) << value;
   appendElement(root, name, val.str());
}
      
void DataArchiver::problemSetup(const ProblemSpecP& params)
{
   ProblemSpecP p = params->findBlock("DataArchiver");
   p->require("filebase", d_filebase);
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1;

   d_currentTimestep = 0;

   d_dir = Dir::create(d_filebase);

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
   out2 << params->getDocument() << endl;
}

void DataArchiver::finalizeTimestep(double time, double delt,
				    const LevelP& level, SchedulerP& sched,
				    DataWarehouseP& /*old_dw*/,
				    DataWarehouseP& new_dw)
{
   int timestep = d_currentTimestep;
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

      appendElement(timeElem, "timestepNumber", timestep-1);
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

	 appendElement(levelElem, "numRegions", level->numRegions());
	 appendElement(levelElem, "totalCells", level->totalCells());
	 Level::const_regionIterator iter;
	 for(iter=level->regionsBegin(); iter != level->regionsEnd(); iter++){
	    const Region* region=*iter;
	    DOM_Element regionElem = doc.createElement("Region");
	    levelElem.appendChild(regionElem);
	    appendElement(regionElem, "id", region->getID());
	    appendElement(regionElem, "lowIndex", region->getCellLowIndex());
	    appendElement(regionElem, "highIndex", region->getCellHighIndex());
	    appendElement(regionElem, "resolution", region->getNCells());
	    Box box = region->getBox();
	    appendElement(regionElem, "lower", box.lower());
	    appendElement(regionElem, "upper", box.upper());
	    appendElement(regionElem, "totalCells", region->totalCells());
	 }
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

   // Schedule a bunch o tasks - one for each variable, for each region
   // This will need to change for parallel code
   int n=0;
   Level::const_regionIterator iter;
   for(iter=level->regionsBegin(); iter != level->regionsEnd(); iter++){

      const Region* region=*iter;
      for(int i=0;i<vars.size();i++){
	 for(int j=0;j<number[i];j++){
	    Task* t = new Task("DataArchiver::output", region, new_dw, new_dw,
			       this, &DataArchiver::output, timestep,
			       vars[i], j);
	    t->requires(new_dw, vars[i], j, region, Ghost::None);
	    sched->addTask(t);
	    n++;
	 }
      }
   }
   cerr << "Created " << n << " output tasks\n";
}

static DOM_Node findNextNode(const std::string& name, DOM_Node node)
{
  // Iterate through all of the child nodes that have this name
  DOM_Node found_node = node.getNextSibling();

  DOMString search_name(name.c_str());
  while(found_node != 0){
    DOMString node_name = found_node.getNodeName();
    if (search_name.equals(node_name) ) {
      break;
    }
    found_node = found_node.getNextSibling();
  }
  return found_node;
}


static DOM_Node findNode(const std::string &name,DOM_Node node)
{
  // Convert string name to a DOMString;
  
  DOMString search_name(name.c_str());
      
  // Do the child nodes now
  DOM_Node child = node.getFirstChild();
  while (child != 0) {
    DOMString child_name = child.getNodeName();
#if 0
    char *s = child_name.transcode();
    cerr << "child_name=" << s << ", searching for: " << name << "\n";
    delete[] s;
#endif
    if (search_name.equals(child_name) ) {
      return child;
    }
    //DOM_Node tmp = findNode(name,child);
    child = child.getNextSibling();
  }
  DOM_Node unknown;
  return unknown;
}

static DOM_Node findTextNode(DOM_Node node)
{
   for (DOM_Node child = node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 return child;
      }
   }
  DOM_Node unknown;
  return unknown;   
}

void DataArchiver::output(const ProcessorContext*,
			  const Region* region,
			  DataWarehouseP& /*old_dw*/,
			  DataWarehouseP& new_dw,
			  int timestep,
			  const VarLabel* var,
			  int matlIndex)
{
   cerr << "output called on region: " << region->getID() << ", variable: " << var->getName() << ", material: " << matlIndex << " at time: " << timestep << "\n";
   
   ostringstream tname;
   tname << "t" << setw(4) << setfill('0') << timestep;

   Dir tdir = d_dir.getSubdir(tname.str());

   ostringstream lname;
   lname << "l0"; // Hard coded - steve
   Dir ldir = tdir.getSubdir(lname.str());


   ostringstream pname;
   pname << ldir.getName() << "/p" << setw(5) << setfill('0') << d_MpiRank << ".xml";

   ProblemSpecP prob_spec;
   DOM_Document doc;
   string pname_s = pname.str();
   ifstream test(pname_s.c_str());
   if(test){
      // Instantiate the DOM parser.
      DOMParser parser;
      parser.setDoValidation(false);

      DAErrorHandler handler;
      parser.setErrorHandler(&handler);

      // Parse the input file
      // No exceptions just yet, need to add

      parser.parse(pname_s.c_str());

      if(handler.foundError)
	 throw InternalError("Error reading file: "+pname_s);

      // Add the parser contents to the ProblemSpecP d_doc

      prob_spec = new ProblemSpec;
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
   appendElement(pdElem, "region", region->getID());

   // Open the data file
   ostringstream dname;
   dname << ldir.getName() << "/p" << setw(5) << setfill('0') << d_MpiRank << ".data";
   string datafile = dname.str();
   int fd = fd = open(datafile.c_str(), O_WRONLY|O_CREAT, 0666);
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
      char* zero = new char[pad];
      bzero(zero, pad);
      write(fd, zero, pad);
      cur+=pad;
      delete[] zero;
   }
   ASSERTEQ(cur%PADSIZE, 0);
   appendElement(pdElem, "start", cur);

   OutputContext oc(fd, cur);
   new_dw->emit(oc, var, matlIndex, region);
   appendElement(pdElem, "end", oc.cur);
   s = fstat(fd, &st);
   if(s == -1)
      throw ErrnoException("DataArchiver::output (stat call)", errno);
   ASSERTEQ(oc.cur, st.st_size);

   s = close(fd);
   if(s == -1)
      throw ErrnoException("DataArchiver::output (close call)", errno);

   ofstream out(pname_s.c_str());
   out << doc << endl;
}

//
// $Log$
// Revision 1.1  2000/05/15 19:39:35  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
//
