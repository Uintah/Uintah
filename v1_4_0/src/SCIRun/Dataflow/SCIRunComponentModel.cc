/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  SCIRunComponentModel.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Dataflow/SCIRunComponentModel.h>
#include <SCIRun/Dataflow/SCIRunComponentDescription.h>
#include <SCIRun/Dataflow/SCIRunComponentInstance.h>
#include <SCIRun/SCIRunErrorHandler.h>
#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Containers/StringUtil.h>
#include <sci_defs.h>
#include <Core/OS/Dir.h>
#include <Core/Util/scirun_env.h>
#include <Core/Util/soloader.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

namespace SCIRun {
env_map scirunrc;                        // contents of .scirunrc
}

static void postMessage(const string& msg)
{
  cerr << msg << '\n';
}

SCIRunComponentModel::SCIRunComponentModel(SCIRunFramework* framework)
  : ComponentModel("scirun"), framework(framework)
{
  buildComponentList();
  tcl_started=false;
}

SCIRunComponentModel::~SCIRunComponentModel()
{
  destroyComponentList();
}

bool SCIRunComponentModel::haveComponent(const std::string& type)
{
  return components.find(type) != components.end();
}

ComponentInstance* SCIRunComponentModel::createInstance(const std::string& name,
							const std::string& type)
{
  componentDB_type::iterator iter = components.find(type);
  if(iter == components.end())
    return 0;
  SCIRunComponentDescription* cd = iter->second;
  string libname;
  DOM_Node n = cd->getNode();

  string libpath="";
  env_iter envi = scirunrc.find("PACKAGE_LIB_PATH");
  if (envi!=scirunrc.end())
    libpath=(*envi).second;

  if(!tcl_started){
    string lib = "libCore_GuiInterface.so";
    LIBRARY_HANDLE handle = FindLibInPath(lib.c_str(), libpath.c_str());
    if(!handle){
      cerr << "Error opening GuiInterface so!\n";
      return 0;
    }
    string sym = "start_TCL";
    void* start = GetHandleSymbolAddress(handle, sym.c_str());
    if(!start){
      cerr << "Error finding " << sym << " in " << lib << '\n';
      return 0;
    }
    void (*start_fn)() = (void (*)())start;
    (*start_fn)();
    tcl_started=true;
  }

  LIBRARY_HANDLE handle = 0;
  for (DOM_Node child = n.getFirstChild();
       child!=0 && !handle; child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("libraries")){
      for (DOM_Node lib = child.getFirstChild();
	   lib!=0 && !handle; lib=lib.getNextSibling()) {
	DOMString name = lib.getNodeName();
	if (name.equals("library")){
	  for (DOM_Node n = lib.getFirstChild(); child != 0 && !handle;
	       n = n.getNextSibling()) {
	    if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	      DOMString val = child.getNodeValue();
	      char* libname = val.transcode();
	      handle = FindLibInPath(libname, libpath);
	      cerr << "1. Looking for lib: " << libname << " in path " << libpath << '\n';
	    }
	  }
	}
      }
    }
  }
  if(libname != "")
    handle = GetLibraryHandle(libname.c_str());
  vector<string> elems = split_string(cd->getType(), '.');
  cerr << "type=" << cd->getType() << '\n';
  if(!handle){
    vector<string> prefixes;
    prefixes.push_back("Dataflow_Modules");
    prefixes.push_back("Packages_Dataflow_Modules");
    for(vector<string>::iterator prefix = prefixes.begin();
	prefix != prefixes.end() && !handle; prefix++){
      for(int i=1;i<(int)elems.size()-1 && !handle;i++){
	string lname = "";
	for(int j=1;j<=i;j++){
	  lname += "_";
	  lname += elems[j];
	}
	string libname = "lib"+*prefix+lname+".so";
	handle = FindLibInPath(libname, libpath);
	cerr << "2. Looking for lib: " << libname << " in path " << libpath << '\n';
      }
    }
  }
  if(!handle){
    cerr << "Cannot find library for component: " << name << " in path " << libpath << '\n';
    return 0;
  }
  void* maker = 0;
  for(int i=0;i<(int)elems.size() && !maker;i++){
    string makername = "make";
    for(int j=i;j<(int)elems.size();j++){
      makername += "_";
      makername += elems[j];
    }
    cerr << "Looking for maker as symbol " << makername << '\n';
    maker = GetHandleSymbolAddress(handle, makername.c_str());
  }
  if(!maker){
    cerr << "Cannot find symbol for " << cd->getType() << " in library\n";
    return 0;
  }
  Module* (*m)(const std::string&) = (Module* (*)(const std::string&))maker;
  Module* module = (*m)(name);
  SCIRunComponentInstance* ci = new SCIRunComponentInstance(framework, name,
							    type, module);
  return ci;
}

string SCIRunComponentModel::getName() const
{
  return "Dataflow";
}

void SCIRunComponentModel::listAllComponentTypes(vector<ComponentDescription*>& list,
						 bool listInternal)
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    list.push_back(iter->second);
  }
}

void SCIRunComponentModel::buildComponentList()
{
  // Initialize the XML4C system
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :\n"
	 << StrX(toCatch.getMessage()) << endl;
    return;
  }

  destroyComponentList();

  // the format of PACKAGE_PATH is a colon seperated list of paths to the
  // root(s) of package source trees.
  // build the complete package path (var in .scirunrc + default)
  env_iter envi = scirunrc.find(string("PACKAGE_SRC_PATH"));
  string packagePath;
  if (envi!=scirunrc.end())
    packagePath = (*envi).second + ":" + SCIRUN_SRCDIR +"/Packages";
  else
    packagePath = string(SCIRUN_SRCDIR) + "/Packages";

  // the format of LOAD_PACKAGE is a comma seperated list of package names.
  // build the complete list of packages to load
  envi = scirunrc.find(string("LOAD_PACKAGE"));
  string loadPackage;
  if (envi!=scirunrc.end())
    loadPackage = (*envi).second;
  else
    loadPackage = LOAD_PACKAGE;

  while(loadPackage!="") {
    // Strip off the first element, leave the rest for the next
    // iteration.
    const unsigned int firstComma = loadPackage.find(',');
    string packagename;
    if(firstComma < loadPackage.size()) {
      packagename=loadPackage.substr(0,firstComma);
      loadPackage=loadPackage.substr(firstComma+1);
    } else {
      packagename=loadPackage;
      loadPackage="";
    }

    string tmpPath = packagePath;
    string pathElt;

    bool found=false;
    if(packagename == "SCIRun"){
      found=true;
    } else {
      while(tmpPath!="") {
	const unsigned int firstColon = tmpPath.find(':');
	if(firstColon < tmpPath.size()) {
	  pathElt=tmpPath.substr(0,firstColon);
	  tmpPath=tmpPath.substr(firstColon+1);
	} else {
	  pathElt=tmpPath;
	  tmpPath="";
	}
	Dir d(pathElt+"/"+packagename);
	if(d.exists()){
	  found=true;
	  break;
	}
      }
    }

    if (!found){
      postMessage("Unable to load package " + packagename +
		  ":\n - Can't find " + packagename + 
		  " directory in package path\n");
      continue;
    }

    string libprefix;
    string xmldir;
    if(packagename == "SCIRun"){
      libprefix = "";
      xmldir = string(SCIRUN_SRCDIR) + "/Dataflow/XML";
    } else {
      libprefix = "Packages_" + packagename + "_";
      xmldir = pathElt+"/"+packagename+"/Dataflow/XML";
    }

    Dir d(xmldir);
    vector<string> files;
    d.getFilenamesBySuffix(".xml", files);

    if (files.size() == 0) {
      postMessage("Unable to load package " + packagename +
		  ":\n - Couldn't find *.xml in " + xmldir +"\n");
      continue;
    }

    for (vector<string>::iterator iter=files.begin();
	 iter!=files.end();iter++) {
      string filename = *iter;
      
      // Instantiate the DOM parser.
      DOMParser parser;
      parser.setDoValidation(false);
  
      SCIRunErrorHandler handler;
      parser.setErrorHandler(&handler);
  
      string file = xmldir+"/"+filename;
      try {
	parser.parse(file.c_str());
      }  catch (const XMLException& toCatch) {
	std::cerr << "Error during parsing: '" <<
	  file << "'\nException message is:  " <<
	  xmlto_string(toCatch.getMessage());
	handler.foundError=true;
	continue;
      }
  
      DOM_Document doc = parser.getDocument();
      DOM_NodeList list = doc.getElementsByTagName("component");
      int nlist = list.getLength();
      for (int i=0;i<nlist;i++){
	DOM_Node n = list.item(i);
	SCIRunComponentDescription* cd=new SCIRunComponentDescription(this, n, packagename);
	componentDB_type::iterator iter2 = components.find(cd->getType());
	if(iter2 != components.end()){
	  cerr << "Warning: component multiply defined: " << iter2->first << "( in file " << *iter << ")\n";
	} else {
	  if(cd->valid()){
	    components[cd->getType()]=cd;
	  } else {
	    delete cd;
	  }
	}
      }
    }
  }
}

void SCIRunComponentModel::destroyComponentList()
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    delete iter->second;
  }
  components.clear();
}

