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

// PackageDB.cc - Interface to module-finding and loading mechanisms

#include <Core/Util/scirun_env.h>
#ifdef ASSERT
#undef ASSERT
#endif
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/FileUtils.h>
#include <Dataflow/Network/ComponentNode.h>
#include <Dataflow/Network/PackageDBHandler.h>
#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <stdio.h>
#include <iostream>
#include <ctype.h>
#include <string>
#include <vector>
using namespace std;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif

#include <sys/stat.h>
static string SCIRUN_SRCTOP = SRCTOP;   // = INSTALL_DIR/SCIRun/src
static string SCIRUN_OBJTOP = OBJTOP;   // = BUILD_DIR
static string DEFAULT_LOAD_PACKAGE = LOAD_PACKAGE;  // configured packages

#ifdef __APPLE__
static string lib_ext = ".dylib";
#else
static string lib_ext = ".so";
#endif

namespace SCIRun {

  PackageDB* packageDB = 0;
  //env_map scirunrc;                        // contents of .scirunrc

  typedef struct {
    string name;
    std::map<string,ModuleInfo*> modules;
  } category;
  
  typedef struct {
    string name;
    std::map<string,category*> categories;
  } package;

  typedef std::map<int,char*>::iterator char_iter;
  typedef std::map<int,inport_node*>::iterator inport_iter;
  typedef std::map<int,outport_node*>::iterator outport_iter;
  typedef std::map<string,int>::iterator string_iter;
  typedef std::map<string,category*>::iterator category_iter;
  typedef std::map<string,ModuleInfo*>::iterator module_iter;
  typedef std::map<int,package*>::iterator package_iter;
}

using namespace SCIRun;
  
PackageDB::PackageDB(GuiInterface* gui)
  : db_(new Packages), packageList_(0), gui(gui)
{
}

PackageDB::~PackageDB()
{ 
  delete db_; 
}

typedef void (*pkgInitter)(const string& tclPath);

LIBRARY_HANDLE PackageDB::findLibInPath(string lib, string path)
{
  LIBRARY_HANDLE handle;
  string tempPaths = path;
  string dir;

  // try to find the library in the specified path
  while (tempPaths!="") {
    const unsigned int firstColon = tempPaths.find(':');
    if(firstColon < tempPaths.size()) {
      dir=tempPaths.substr(0,firstColon);
      tempPaths=tempPaths.substr(firstColon+1);
    } else {
      dir=tempPaths;
      tempPaths="";
    }

    handle = GetLibraryHandle((dir+"/"+lib).c_str());
    if (handle)
      return handle;
  }

  // if not yet found, try to find it in the rpath 
  // or the LD_LIBRARY_PATH (last resort)
  handle = GetLibraryHandle(lib.c_str());
  return handle;
}

bool PackageDB::findMaker(ModuleInfo* moduleInfo)
{
  string cat_bname, pak_bname;
  if(moduleInfo->packageName == "SCIRun") {
    cat_bname = "Dataflow_Modules_";
    pak_bname = "Dataflow";
  } else {
    cat_bname = "Packages_" + moduleInfo->packageName + "_Dataflow_Modules_";
    pak_bname = "Packages_" + moduleInfo->packageName + "_Dataflow";
  }

  string libpath="";
  env_iter envi = scirunrc.find("PACKAGE_LIB_PATH");
  if (envi!=scirunrc.end())
    libpath=(*envi).second;

  // try the large version of the shared library
  string libname = "lib" + pak_bname +lib_ext;
  LIBRARY_HANDLE package_so = findLibInPath(libname,libpath);
  string package_error;
  if (!package_so)
    package_error = SOError();

  // try the small version of the shared library
  string cat_name = moduleInfo->categoryName;
  if (cat_name.substr(0, 6) == "Fields") { cat_name = "Fields"; }
  libname = "lib" + cat_bname + cat_name + lib_ext;
  LIBRARY_HANDLE category_so = findLibInPath(libname,libpath);
  string category_error;
  if (!category_so)
    category_error = SOError();

  if (!category_so && !package_so) {

    string msg = "Unable to load all of package '" + moduleInfo->packageName +
      "' (category '" + moduleInfo->categoryName + "' failed) :\n - " +
      package_error + "\n - " + category_error;

    if(gui){
      gui->postMessage( msg );
      gui->execute("update idletasks");
    } else {
      cerr << msg << "\n";
    }
    return false;
  }

  string makename = "make_" + moduleInfo->moduleName;
  if (category_so)
    moduleInfo->maker = 
      (ModuleMaker)GetHandleSymbolAddress(category_so,makename.c_str());
  if (!moduleInfo->maker && package_so)
    moduleInfo->maker = 
      (ModuleMaker)GetHandleSymbolAddress(package_so,makename.c_str());
  if (!moduleInfo->maker) {
    if(gui){
      gui->postMessage("Unable to load module '" + moduleInfo->moduleName +
		       "' :\n - can't find symbol 'make_" +
		       moduleInfo->moduleName + "'\n");
      gui->execute("update idletasks");
    } else {
      cerr << "Unable to load module '" << moduleInfo->moduleName
	   << "' :\n - can't find symbol 'make_" << moduleInfo->moduleName
	   << "'\n";
    }
    return false;
  }
  return true;
}


void PackageDB::loadPackage(bool resolve)
{
  string loadPackage;
  string result;
  std::map<int,package*> packages;
  package* new_package = 0;
  category* new_category = 0;
  ModuleInfo* new_module = 0;
  module_iter mi;
  category_iter ci;
  package_iter pi;
  string packageElt;
  component_node* node = 0;
  int mod_count = 0;
  string notset(NOT_SET);
  string packagePath;

  if(gui)
    gui->postMessage("Loading packages, please wait...", false);
  else
    cerr << "Loading packages...\n";

#ifdef __APPLE__
  // A hack around a gcc (apple) init bug
  GetLibrarySymbolAddress( "lib/libCore_Datatypes.dylib",
			   "__ZN6SCIRun10CurveFieldINS_6TensorEE2ioERNS_9PiostreamE");
#endif

  // the format of PACKAGE_PATH is a colon seperated list of paths to the
  // root(s) of package source trees.
  // build the complete package path (var in .scirunrc + default)
  env_iter envi = scirunrc.find(string("PACKAGE_SRC_PATH"));
  if (envi!=scirunrc.end())
    packagePath = (*envi).second + ":" + SCIRUN_SRCTOP +"/Packages";
  else
    packagePath = SCIRUN_SRCTOP + "/Packages";

  // the format of LOAD_PACKAGE is a comma seperated list of package names.
  // build the complete list of packages to load
  envi = scirunrc.find(string("LOAD_PACKAGE"));
  if (envi!=scirunrc.end())
    loadPackage = (*envi).second;
  else
    loadPackage = DEFAULT_LOAD_PACKAGE;

  while(loadPackage!="") {
    // Strip off the first element, leave the rest for the next
    // iteration.
    const unsigned int firstComma = loadPackage.find(',');
    if(firstComma < loadPackage.size()) {
      packageElt=loadPackage.substr(0,firstComma);
      loadPackage=loadPackage.substr(firstComma+1);
    } else {
      packageElt=loadPackage;
      loadPackage="";
    }

    string tmpPath = packagePath;
    string pathElt;

    for (;tmpPath!="";) {
      if (packageElt=="SCIRun") {
	tmpPath = "found";
	break;
      }
      const unsigned int firstColon = tmpPath.find(':');
      if(firstColon < tmpPath.size()) {
	pathElt=tmpPath.substr(0,firstColon);
	tmpPath=tmpPath.substr(firstColon+1);
      } else {
	pathElt=tmpPath;
	tmpPath="";
      }
      
      struct stat buf;
      lstat((pathElt+"/"+packageElt).c_str(),&buf);
      if (S_ISDIR(buf.st_mode)) {
	tmpPath = "found";
	break;
      }
    }

    if (tmpPath=="") {
      if(gui){
	gui->postMessage("Unable to load package " + packageElt +
			 ":\n - Can't find " + packageElt + 
			 " directory in package path");
      } else {
	cerr << "Unable to load package " << packageElt
	     << ":\n - Can't find " << packageElt
	     << " directory in package path\n";
      }
      continue;
    }

    do_command("lappend auto_path "+pathElt+"/"+packageElt+"/Dataflow/GUI");

    string bname = packageElt;
    string pname = packageElt;
    string xmldir;
    
    if(bname == "SCIRun") {
      bname = "";
      pname = "SCIRun";
      xmldir = SCIRUN_SRCTOP + "/Dataflow/XML";
      do_command(string("lappend auto_path ")+SCIRUN_SRCTOP+
		 "/Dataflow/GUI");
    } else {
      bname = "Packages_" + bname + "_";
      xmldir = pathElt+"/"+packageElt+"/Dataflow/XML";
      do_command(string("lappend auto_path ")+pathElt+"/"+packageElt+
		 "/Dataflow/GUI");
    }
    std::map<int,char*>* files;
    files = GetFilenamesEndingWith((char*)xmldir.c_str(),".xml");

    if (!files) {
      if(gui)
	gui->postMessage("Unable to load package " + pname +
			 ":\n - Couldn't find *.xml in " + xmldir );
      continue;
    }

    new_package = new package;
    new_package->name=pname;
    packages.insert(std::pair<int,
		    package*>(packages.size(),new_package));

    mod_count += files->size();

    for (char_iter i=files->begin();
	 i!=files->end();
	 i++) {
      if (node) DestroyComponentNode(node);
      node = CreateComponentNode(1);
      ReadComponentNodeFromFile(node,(xmldir+"/"+(*i).second).c_str(), gui);

      if (notset==node->name||notset==node->category) continue;

      ci = new_package->categories.find(string(node->category));
      if (ci==new_package->categories.end()) {
	new_category = new category;
	new_category->name = string(node->category);
	new_package->categories.insert(std::pair<string,
	  category*>(new_category->name,new_category));
	ci = new_package->categories.find(string(new_category->name));
      }
      
      mi = (*ci).second->modules.find(string(node->name));
      if (mi==(*ci).second->modules.end()) {
	IPortInfo* ipinfo;
	OPortInfo* opinfo;
	new_module = new ModuleInfo;
	new_module->moduleName = node->name;
	new_module->categoryName = node->category;
	new_module->packageName = pname;
	new_module->help_description = node->overview->description;
	new_module->maker = 0;
	new_module->uiFile = "not currently used";
	new_module->iports = scinew std::map<int,IPortInfo*>;
	new_module->oports = scinew std::map<int,OPortInfo*>;
	new_module->lastportdynamic = node->io->lastportdynamic;
	for (inport_iter i1 = node->io->inports->begin();
	     i1!=node->io->inports->end();
	     i1++) {
	  ipinfo = scinew IPortInfo;
	  ipinfo->name = string(((*i1).second)->name);
	  ipinfo->datatype = string(((*i1).second)->datatype);
	  ipinfo->maker = (iport_maker)0;
	  new_module->iports->insert(
	    std::pair<int,IPortInfo*>(new_module->iports->size(),
					    ipinfo));
	}
	for (outport_iter i2 = node->io->outports->begin();
	     i2!=node->io->outports->end();
	     i2++) {
	  opinfo = scinew OPortInfo;
	  opinfo->name = string(((*i2).second)->name);
	  opinfo->datatype = string(((*i2).second)->datatype);
	  opinfo->maker = (oport_maker)0;
	  new_module->oports->insert(
	    std::pair<int,OPortInfo*>(new_module->oports->size(),
					    opinfo));
	}
	(*ci).second->modules.insert(std::pair<string,
	   ModuleInfo*>(string(new_module->moduleName),new_module));
      }
    }
  }
  if (gui && !getenv("SCI_NOSPLASH"))
  {
    gui->execute("showSpash " + to_string(mod_count) + " """ +
		 string(SCIRUN_SRCTOP) + "/main/scisplash.ppm""");
  }
  int index = 0;
  int numreg;
  
  for (pi = packages.begin();
       pi!=packages.end();
       pi++) {
    
    numreg = 0;
    
    string pname = (*pi).second->name;

    if(gui){
      gui->postMessage("Loading package '" + pname + "'", false);
      if (!getenv("SCI_NOSPLASH"))
      {
	gui->execute(".splash.fb configure -labeltext {Loading package: " +
		     pname + " }");
	gui->eval("update idletasks",result);
      }
    } else {
      cerr << "Loading package '" << pname << "'\n";
    }

    for (ci = (*pi).second->categories.begin();
	 ci!=(*pi).second->categories.end();
	 ci++) {
      for (mi = (*ci).second->modules.begin();
	   mi!=(*ci).second->modules.end();
	   mi++) {
	if(resolve){
	  if(findMaker((*mi).second)){
	    registerModule((*mi).second);
	    numreg++;
	  } else {
	    string mname = (*mi).second->moduleName;
	    if(gui){
	      gui->postMessage("Unable to load module '" + mname +
			       "' :\n - can't find symbol 'make_" + mname + "'\n");
	      gui->execute("update idletasks");
	    } else {
	      cerr << "Unable to load module '" << mname
		   << "' :\n - can't find symbol 'make_" << mname << "'\n";
	    }
	  }
	} else {
	  numreg++;
	  registerModule((*mi).second);
	}
	if (gui && !getenv("SCI_NOSPLASH"))
	{
	  gui->execute("if [winfo exists .splash.fb] "
		       "{.splash.fb step; update idletasks}");
	}
      }
    }
    
    if (numreg) {
      if(gui)
	gui->execute("createPackageMenu " + to_string(index++));
    } else {
      if(gui)
	gui->postMessage("Unable to load package " + pname + ":\n"
			 " - could not find any valid modules.");
      else
	cerr << "Unable to load package " << pname
	     <<":\n - could not find any valid modules.\n";
    }
  }

  if(gui){
    gui->postMessage("\nFinished loading packages.",false);
    if (!getenv("SCI_NOSPLASH"))
    {
      gui->execute("if [winfo exists .splash] {destroy .splash}");
      gui->eval("update idletasks",result);
    }
  } else {
    cerr << "Finished loading packages\n";
  }
}
  
void PackageDB::registerModule(ModuleInfo* info) {
  Package* package;
  if(!db_->lookup(info->packageName,package))
    {
      db_->insert(info->packageName,package=new Package);
      packageList_.push_back( info->packageName );
    }
  
  Category* category;
  if(!package->lookup(info->categoryName,category))
    package->insert(info->categoryName,category=new Category);
  
  ModuleInfo* moduleInfo;
  if(!category->lookup(info->moduleName,moduleInfo)) {
    moduleInfo=new ModuleInfo;
    category->insert(info->moduleName,info);
  } else cerr << "WARNING: Overriding multiply registered module "
	      << info->packageName << "." << info->categoryName << "."
	      << info->moduleName << "\n";  
}
 
void PackageDB::createAlias(const string& fromPackageName,
			    const string& fromCategoryName,
			    const string& fromModuleName,
			    const string&,// toPackageName,
			    const string&,// toCategoryName,
			    const string&)// toModuleName)
{
  Package* package;
  if(!db_->lookup(fromPackageName,package)) {
    gui->postMessage("Warning: creating an alias from a nonexistant package "+fromPackageName+" (ignored)");
    return;
  }
  
  Category* category;
  if(!package->lookup(fromCategoryName,category)) {
    gui->postMessage("Warning: creating an alias from a nonexistant category "+fromPackageName+"."+fromCategoryName+" (ignored)");
    return;
  }
  
  ModuleInfo* moduleInfo;
  if(!category->lookup(fromModuleName,moduleInfo)) {
    gui->postMessage("Warning: creating an alias from a nonexistant module "+fromPackageName+"."+fromCategoryName+"."+fromModuleName+" (ignored)");
    return;
  }
  registerModule(moduleInfo);
}
 
Module* PackageDB::instantiateModule(const string& packageName,
				     const string& categoryName,
				     const string& moduleName,
				     const string& instanceName)
{
  Package* package;
  if(!db_->lookup(packageName,package)) {
    cerr << "ERROR: Instantiating from nonexistant package " << packageName 
	 << "\n";
    return 0;
  }
  
  Category* category;
  if(!package->lookup(categoryName,category)) {
    cerr << "ERROR: Instantiating from nonexistant category " << packageName
	 << "." << categoryName << "\n";
    return 0;
  }
  
  ModuleInfo* moduleInfo;
  if(!category->lookup(moduleName,moduleInfo)) {
    cerr << "ERROR: Instantiating nonexistant module " << packageName 
	 << "." << categoryName << "." << moduleName << "\n";
    return 0;
  }
  
#if 0
  // This was McQ's somewhat silly replacement for TCL's tclIndex/auto_path
  // mechanism.  The idea was that there would be a path in the index.cc
  // that pointed to a TCL file to source before instantiating a module
  // of some particular class for the frist time -- sortof a TCL-end class
  // constructor for the module's class.
  // Steve understandably doesn't like new, fragile mechanisms where
  // perfectly good old, traditional ones already exist, so he if0'd this
  // away and added the "lappend auto_path" at package-load-time, above.
  // This code is still here 'cause Some Day it might be nice to allow the
  // source of the TCL files to be stored in the .so (as strings) and eval'd
  // here.  This was the "faraway vision" that drove me to do things this way
  // in the first place, but since that vision seems to have stalled
  // indefinately in lieu of Useful Work, there's no reason not to use
  // auto_path (except that it produces yet one more file to maintain).  And
  // auto_path is useful if you write global f'ns and want to use them in lots
  // of your modules -- auto_path nicely handles this whereas the code below
  // doesn't handle it at all.
  // Some day it might be nice to actually achieve the "package is one .so
  // and that's all" vision, but not today.  :)
  //                                                      -mcq 99/10/6
  
  if(moduleInfo->uiFile!="") {
    string result;
    if(!TCL::eval("source " + moduleInfo->uiFile , result)) {
      cerr << "Can't source UI file " << moduleInfo->uiFile << "...\n";
      cerr << "  TCL Error: " << result << "\n";
    }
    moduleInfo->uiFile="";                       // Don't do it again
  }
#endif

  if(!moduleInfo->maker){
    if(!findMaker(moduleInfo)){
      cerr << "ERROR: Cannot find maker for module: " << packageName 
	   << "." << categoryName << "." << moduleName << "\n";
      return 0;
    }
  }

  GuiContext* module_context = gui->createContext(instanceName);
  Module *module = (moduleInfo->maker)(module_context);
  if(!module)
    return 0;
  
  // Some modules may already know their package and category.
  // If this module doesn't, then set it's package and category here.
  string unknown("unknown");
  if (unknown == module->packageName)
    module->packageName=packageName;
  if (unknown == module->categoryName)
    module->categoryName=categoryName;

  if (moduleInfo->help_description != "(null string)")
  {
    module->description = moduleInfo->help_description;
  }
  else
  {
    module->description = "No help found for this module.";
  }

  // copy other fields 
  module->lastportdynamic = moduleInfo->lastportdynamic;
  
  return module;
}
 
bool PackageDB::haveModule(const string& packageName,
			   const string& categoryName,
			   const string& moduleName) const
{
  Package* package;
  if(!db_->lookup(packageName,package))
    return false;
  
  Category* category;
  if(!package->lookup(categoryName,category))
    return false;
  
  ModuleInfo* moduleInfo;
  if(!category->lookup(moduleName,moduleInfo))
    return false;

  return true;
}
 
vector<string>
PackageDB::packageNames(void) const
{
  // packageList_ is used to keep a list of the packages 
  // that are in this PSE IN THE ORDER THAT THEY ARE SPECIFIED
  // by the user in the Makefile (for main.cc) or in their
  // environment.
  
  return packageList_;
}

vector<string>
PackageDB::categoryNames(const string& packageName) const
{
  Package* package;
  if(!db_->lookup(packageName, package)){
    cerr << "WARNING: Unknown package " << packageName << "\n";
    vector<string> result(0);
    return result;
  }
  vector<string> result(package->size());
  PackageIter iter(package);
  int i=0;
  for(iter.first();iter.ok();++iter) result[i++]=iter.get_key();
  return result;
}


string
PackageDB::getCategoryName(const string &packName,
			   const string &catName,
			   const string &modName)
{
  Package *package;
  if (!db_->lookup(packName, package)){
    cerr << "WARNING: Unknown package " << packName << "\n";
    return catName;
  }

  Category *category;
  ModuleInfo* modinfo;
  if (package->lookup(catName, category) &&
      category->lookup(modName, modinfo))
  {
    // Original category was fine, just return that.
    return catName;
  }

  PackageIter iter(package);
  for (iter.first(); iter.ok();++iter)
  {
    if (iter.get_data()->lookup(modName, modinfo))
    {
      string newcatname = iter.get_key();
      if (newcatname != catName &&
	  !(packName == "SCIRun" && catName == "Fields"))
      {
	cerr << "WARNING: Remapping " <<
	  packName << "_" << catName << "_" << modName << " to " <<
	  packName << "_" << newcatname << "_" << modName << "\n";
      }
      return newcatname;
    }
  }
  return catName;
}
 

vector<string>
PackageDB::moduleNames(const string& packageName,
		       const string& categoryName) const
{
  Package* package;
  if(!db_->lookup(packageName, package)){
    cerr << "WARNING: Unknown package " << packageName << "\n";
    vector<string> result(0);
    return result;
  }

  Category* category;
  if(!package->lookup(categoryName, category)){
    cerr << "WARNING: Unknown category " << packageName << "."
	 << categoryName << "\n";
    vector<string> result(0);
    return result;
  }
  vector<string> result(category->size());
  CategoryIter iter(category);
  int i=0;
  for(iter.first();iter.ok();++iter) 
    result[i++]=iter.get_key();
  return result;
}

void PackageDB::setGui(GuiInterface* gui)
{
  this->gui=gui;
  for(vector<string>::iterator iter = delayed_commands.begin();
      iter != delayed_commands.end(); ++iter){
    gui->execute(*iter);
  }
  delayed_commands.clear();
}

void PackageDB::do_command(const string& command)
{
  if(gui)
    gui->execute(command);
  else
    delayed_commands.push_back(command);
}

ModuleInfo* PackageDB::GetModuleInfo(const string& name, const string& catname,
				     const string& packname)
{
  Package* package;
  if (!db_->lookup(packname,package))
    return 0;

  Category* category;
  if (!package->lookup(catname,category))
    return 0;

  ModuleInfo* info;
  if (category->lookup(name,info))
    return info;
  return 0;
}


