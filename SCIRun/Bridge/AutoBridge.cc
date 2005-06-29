/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


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
#include <Core/Util/Environment.h>

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

string readMetaFile(string componentN, string extN) {
  // Initialize the XML4C system
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :\n"
	      << StrX(toCatch.getMessage()) << endl;
    return "";
  }
 
  std::string srcdir(SCIRun::sci_getenv("SCIRUN_SRCDIR"));
 
  std::vector<std::string> sArray;
  sArray.resize(2);
  sArray[0] = srcdir + "/CCA/Components/xml";
  sArray[1] = srcdir + "/CCA/Components/BabelTest/xml";

  for (std::vector<std::string>::iterator it = sArray.begin(); it != sArray.end(); it++) {
    Dir d(*it);
    std::vector<std::string> files;
    d.getFilenamesBySuffix(".xml", files);
    
    for(std::vector<std::string>::iterator iter = files.begin();
        iter != files.end(); iter++) {
      std::string& xmlfile = *iter;
      std::cerr << "Auto Bridge: Looking at file" << xmlfile << std::endl;
      std::string file(*it+"/"+xmlfile);
                                                                                                                                      
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
      
      DOMDocument* document = parser.getDocument();
      
      // Get a list of the library nodes.  Traverse the list and read component
      // elements at each list node.
      DOMNodeList* libraries
	= document->getElementsByTagName(to_xml_ch_ptr("library"));
      
      for (unsigned int i = 0; i < libraries->getLength(); i++) {
	DOMElement *library = static_cast<DOMElement *>(libraries->item(i));
	// Read the library name
	std::string library_name(to_char_ptr(library->getAttribute(to_xml_ch_ptr("name"))));
	
	// Get the list of components.
	DOMNodeList* comps
	  = library->getElementsByTagName(to_xml_ch_ptr("component"));
	for (unsigned int j = 0; j < comps->getLength(); j++) {
	  // Read the component name
	  DOMElement *component = static_cast<DOMElement *>(comps->item(j));
	  std::string
	    component_name(to_char_ptr(component->getAttribute(to_xml_ch_ptr("name"))));

	  if(componentN != component_name) {
	    continue;
	  }
	  
	  DOMNodeList* ifaces 
	    = library->getElementsByTagName(to_xml_ch_ptr("interface"));
	  if(ifaces->getLength() != 1) {
	    std::cerr << "Error during parsing: '" << file << "'\n";
	    return "";
	  }                 
	  DOMElement *interface = static_cast<DOMElement *>(ifaces->item(0));
	  std::string
	    file_name(to_char_ptr(interface->getAttribute(to_xml_ch_ptr("file"))));
          std::string wholename(srcdir + "/" + file_name); 
	  return wholename; 
	}
      }
    }
  }
  return "";
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
  string cCCA; //Used so that we read xml data only from components that have .cca files 
 
  if(modelFrom == "babel") {
    if(modelTo != "cca") cCCA = cFrom;
    cFrom = cFrom.substr(0,cFrom.find(".")); //Babel xxx.Com 
  } else if(modelFrom == "cca") {
    cCCA = cFrom;
    cFrom = cFrom.substr(cFrom.find(".")+1); //CCA SCIRun.xxx
  } else if(modelFrom == "dataflow") {
    cFrom = cFrom.substr(cFrom.rfind(".")+1); //SCIRun.yyy.xxx
  } else if(modelFrom == "vtk") {
    cFrom = cFrom.substr(cFrom.rfind(".")+1); //Vtk.xxx
  } else if(modelFrom == "tao") {
    cFrom = cFrom.substr(cFrom.rfind(".")+1); //Tao.xxx
  } else {}
  

  if(modelTo == "babel") {
    if(modelFrom != "cca") cCCA = cTo;
    cTo = cTo.substr(0,cTo.find(".")); //Babel xxx.Com 
  } else if(modelTo == "cca") {
    cCCA = cTo;
    cTo = cTo.substr(cTo.find(".")+1); //CCA SCIRun.xxx
  } else if(modelTo == "dataflow") {
    cTo = cTo.substr(cTo.rfind(".")+1); //SCIRun.yyy.xxx
  } else if(modelTo == "vtk") {
    cTo = cTo.substr(cTo.rfind(".")+1); //Vtk.xxx
  } else if(modelTo == "tao") {
    cTo = cTo.substr(cTo.rfind(".")+1); //Tao.xxx
  } else {}

  string name = cFrom+"__"+cTo;

  //Check runtime cache
  if(runC.find(name) != runC.end()) return name;

  //read cca file to find .sidl file
  cerr << "read cca file to find .sidl file -- " << cFrom << ".cca\n";
  string sidlfile = readMetaFile(cCCA,"cca");
  if(sidlfile=="") {cerr << "ERROR... exiting...\n"; return "";} 

  //determine right plugin
  string hdrplugin;
  string plugin;
  string util;
  string srcdir = sci_getenv("SCIRUN_SRCDIR");
  string templatedir(srcdir + string("/Core/CCA/tools/scim/template/"));

  if((modelFrom == "babel")&&(modelTo == "cca")) {
    plugin = templatedir + string("BabeltoCCA.erb");
    hdrplugin = templatedir + string("BabeltoCCA.hdr.erb"); 
    util = templatedir + string("CCAtoBabel.util.rb"); 
  } else if((modelFrom == "cca")&&(modelTo == "babel")) {
    plugin = templatedir + string("CCAtoBabel.erb"); 
    hdrplugin = templatedir + string("CCAtoBabel.hdr.erb"); 
    util = templatedir + string("CCAtoBabel.util.rb"); 
  } else if((modelFrom == "dataflow")&&(modelTo == "cca")) {
    plugin = templatedir + string("BabeltoCCA.erb"); 
  } else if((modelFrom == "cca")&&(modelTo == "dataflow")) {
    plugin = templatedir + string("CCAtoDataflow.erb"); 
  } else if((modelFrom == "babel")&&(modelTo == "vtk")) {
    plugin = templatedir + string("BabeltoVtk.erb"); 
    hdrplugin = templatedir + string("BabeltoVtk.hdr.erb"); ;
    util = templatedir + string("BabeltoVtk.util.rb"); 
  } else if((modelFrom == "cca")&&(modelTo == "tao")) {
    plugin = templatedir + string("CCAtoTao.erb"); 
    hdrplugin = templatedir + string("CCAtoTao.hdr.erb"); 
    util = templatedir + string("CCAtoBabel.util.rb"); 
  }
  else {}


  //use portxml + plugin in strauss to generate bridge
  cerr << "use portxml + plugin in strauss to generate bridge\n";
  string straussout = COMPILEDIR+"/"+name;
  Strauss* strauss = new Strauss(plugin,hdrplugin,sidlfile,straussout+".h",straussout+".cc",util);
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
  delete strauss;

  //compile bridge
  cerr << "compile bridge\n";
  string execline = "cd "+COMPILEDIR+" && gmake && cd ..";
  status = sci_system(execline.c_str());
  if(status!=0) {
    execline = "rm -f "+COMPILEDIR+"/"+name+".*"; 
    //sci_system(execline.c_str());
    cerr << "**** gmake was unsuccessful\n";
    return "";
  }

  //add to runtime cache table
  runC.insert(name);

  return name; 
}

bool AutoBridge::canBridge(PortInstance* pr1, PortInstance* pr2)
{
  std::string t1 = "." + pr1->getType();
  std::string t2 = "." + pr2->getType();
  std::cerr << "Going with " << t1 << " and " << t2 << "\n"; 
  if( pr1->portType()!=pr2->portType() && 
      t1.substr(t1.rfind("."),t1.size()) == t2.substr(t2.rfind("."),t2.size()) )
    return true;
  //For Vtk
  if( pr1->portType()!=pr2->portType() &&
      (pr1->getModel() == "vtk" || pr2->getModel() == "vtk") )
    return true;

  if(pr1->getModel() == "tao" || pr2->getModel() == "tao")
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
