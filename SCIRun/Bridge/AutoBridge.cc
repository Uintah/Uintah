
#include <SCIRun/Bridge/AutoBridge.h> 

#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Util/sci_system.h>

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

std::string AutoBridge::execute(std::string cFrom, std::string cTo) 
{
  int status = 0;

  //Note: limited to (Babel2CCA)
  cFrom = cFrom.substr(0,cFrom.find(".")); //Babel xxx.Com    
  cTo = cTo.substr(cTo.find(".")+1); //CCA SCIRun.xxx

  //read cca file to find .sidl file
  cerr << "read cca file to find .sidl file -- " << cFrom << ".cca\n";
  string sidlfile = readMetaFile(cTo,"cca");
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

  //use portxml + plugin in strauss to generate bridge
  cerr << "use portxml + plugin in strauss to generate bridge\n";
  plugin = "/home/sci/damevski/testprogs/rubytm/BabeltoCCA.erb";
  string name = cFrom+"__"+cTo;
  string straussout = "on-the-fly-libs/"+name;
  execline = "strauss -p " + plugin + " -o " + straussout + " working.kwai";
  cerr << execline << "\n";
  status = sci_system(execline.c_str());
  if(status!=0) {
    cerr << "**** strauss was unsuccessful\n";
    return "";
  }

  //compile bridge
  cerr << "compile bridge\n";
  execline = "cd on-the-fly-libs && gmake && cd ..";
  status = sci_system(execline.c_str());
  if(status!=0) {
    execline = "rm -f on-the-fly-libs/"+name+".*"; 
    //sci_system(execline.c_str());
    cerr << "**** gmake was unsuccessful\n";
    return "";
  }

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

