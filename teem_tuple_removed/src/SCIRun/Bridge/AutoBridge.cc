
#include <SCIRun/Bridge/AutoBridge.h> 

#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Util/sci_system.h>
#include <Core/OS/Dir.h>

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

using namespace SCIRun;
using namespace std;

#define COMPILEDIR std::string("on-the-fly-libs")

string readMetaFile(string component, string ext) {
  // Initialize the XML4C system
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :\n"
	      << StrX(toCatch.getMessage()) << endl;
    return "";
  }
  
  string component_path ="../src/CCA/Components/xml:../src/CCA/Components/BabelTest/xml";
  string file;
  while(component_path != ""){
    unsigned int firstColon = component_path.find(':');
    string dir;
    if(firstColon < component_path.size()){
      dir=component_path.substr(0, firstColon);
      component_path = component_path.substr(firstColon+1);
    } else {
      dir = component_path;
      component_path="";
    }

    file = dir+"/"+component+"."+ext;
    struct stat s;
    if( stat( file.c_str(), &s ) == -1 ) {
      file = "";
    } else {
      break;
    }
  } 
  if(file == "") { cerr << "Meta file does not exist\n"; return ""; }
  
  // Instantiate the DOM parser.
  XercesDOMParser parser;
  parser.setDoValidation(false);
  
  SCIRunErrorHandler handler;
  parser.setErrorHandler(&handler);
  
  try {
    parser.parse(file.c_str());
  }  catch (const XMLException& toCatch) {
    std::cerr << "Error during parsing: '" <<
      file << "'\nException message is:  " <<
      xmlto_string(toCatch.getMessage()) << '\n';
    handler.foundError=true;
    return "";
  }
  
  DOMDocument* doc = parser.getDocument();
  DOMNodeList* list = doc->getElementsByTagName(to_xml_ch_ptr("sidl"));
  int nlist = list->getLength();
  if(nlist == 0){
    cerr << "WARNING: file " << file << " does not contain a sidl file reference!\n";
  }
  DOMNode* d = list->item(0);
  DOMNode* name = d->getAttributes()->getNamedItem(to_xml_ch_ptr("source"));
  if (name==0) {
    cout << "ERROR: Component has no name." << endl;
    return "";
  } else {
    return (to_char_ptr(name->getNodeValue()));
  } 
}

AutoBridge::AutoBridge() 
{
  //populate oldB
  Dir d(COMPILEDIR);
  vector<string> files;
  d.getFilenamesBySuffix(".so", files);
  for(vector<string>::iterator iter = files.begin();
      iter != files.end(); iter++){
    string file = (*iter).substr(0,(*iter).size()-3);
    //cerr << "Found FILE " << file << "\n";
    oldB.insert(file);
  }
  
}

AutoBridge::~AutoBridge() 
{
}

std::string AutoBridge::genBridge(std::string modelFrom, std::string cFrom, std::string modelTo, std::string cTo) 
{
  int status = 0;
  string cCCA; //Used so that we read xml data only from cca components
 
  if(modelFrom == "babel") {
    cFrom = cFrom.substr(0,cFrom.find(".")); //Babel xxx.Com 
  } else if(modelFrom == "cca") {
    cFrom = cFrom.substr(cFrom.find(".")+1); //CCA SCIRun.xxx
    cCCA = cFrom;
  }
  else {}
  
  
  if(modelTo == "babel") {
    cTo = cTo.substr(0,cTo.find(".")); //Babel xxx.Com 
  } else if(modelTo == "cca") {
    cTo = cTo.substr(cTo.find(".")+1); //CCA SCIRun.xxx
    cCCA = cTo;
  }  
  else {}

  string name = cFrom+"__"+cTo;

  //Check runtime cache
  if(runC.find(name) != runC.end()) return name;

  //read cca file to find .sidl file
  cerr << "read cca file to find .sidl file -- " << cFrom << ".cca\n";
  string sidlfile = readMetaFile(cCCA,"cca");
  if(sidlfile=="") {cerr << "ERROR... exiting...\n"; return "";} 

  //use .sidl file in kwai to generate portxml
  cerr << "use .sidl file in kwai to generate portxml\n";
  string execline = "kwai -o working.kwai " + sidlfile;
  status = sci_system(execline.c_str());
  if(status!=0) {
    cerr << "**** kwai was unsuccessful\n";
    return "";
  }

  //determine right plugin
  string plugin;
  if((modelFrom == "babel")&&(modelTo == "cca")) {
    plugin = "../src/Core/CCA/tools/strauss/ruby/BabeltoCCA.erb";
  } else if((modelFrom == "cca")&&(modelTo == "babel")) {
    plugin = "../src/Core/CCA/tools/strauss/ruby/CCAtoBabel.erb";
  }  
  else {}


  //use portxml + plugin in strauss to generate bridge
  cerr << "use portxml + plugin in strauss to generate bridge\n";
  string straussout = COMPILEDIR+"/"+name;
  Strauss* strauss = new Strauss(plugin,"working.kwai",straussout+".h",straussout+".cc");
  status = strauss->emit();
  if(status!=0) {
    cerr << "**** strauss was unsuccessful\n";
    return "";
  }
  //check if a copy exists in oldB, compare CRC to see if it is the right one
  if(oldB.find(name) != oldB.end()) {
    if(isSameFile(name,strauss)) 
      return name;
  }
  //a new bridge, write into files
  strauss->commitToFiles();
  delete strauss;

  //compile bridge
  cerr << "compile bridge\n";
  execline = "cd "+COMPILEDIR+" && gmake && cd ..";
  status = sci_system(execline.c_str());
  if(status!=0) {
    execline = "rm -f "+COMPILEDIR+"/"+name+".*"; 
    sci_system(execline.c_str());
    cerr << "**** gmake was unsuccessful\n";
    return "";
  }

  //add to runtime cache table
  runC.insert(name);

  return name; 
}

bool AutoBridge::canBridge(PortInstance* pr1, PortInstance* pr2)
{
  std::string t1 = pr1->getType();
  std::string t2 = pr2->getType();
  if( pr1->portType()!=pr2->portType() && 
       t1.substr(t1.rfind("."),t1.size()) == t2.substr(t2.rfind("."),t2.size()) )
    return true;
  return false;
}

bool AutoBridge::isSameFile(std::string name, Strauss* strauss)
{
  std::ostringstream impl_crc, hdr_crc;
  impl_crc << strauss->getImplCRC();
  hdr_crc << strauss->getHdrCRC();
  
  char* buf = new char[513];
  string file = COMPILEDIR + "/" + name;

  ifstream ccfile((file+".cc").c_str());
  if(!ccfile) return false;
  ccfile.getline(buf, 512);  
  string cc_sbuf(buf);
  cc_sbuf = cc_sbuf.substr(cc_sbuf.find("=")+1,cc_sbuf.size()-1);
  
  ifstream hfile((file+".h").c_str());
  if(!hfile) return false;
  hfile.getline(buf, 512);  
  string h_sbuf(buf);
  h_sbuf = h_sbuf.substr(h_sbuf.find("=")+1,h_sbuf.size()-1);
  
  delete buf;
  cerr << "Comparing hdr old=" << h_sbuf << " new=" << hdr_crc.str() << "\n";
  cerr << "Comparing impl old=" << cc_sbuf << " new=" << impl_crc.str() << "\n";
  if((h_sbuf == hdr_crc.str())&&(cc_sbuf == impl_crc.str())) 
    return true;
  
  return false;
}
