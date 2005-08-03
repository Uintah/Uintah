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


// PackageDB.cc - Interface to module-finding and loading mechanisms

#ifdef ASSERT
#  undef ASSERT
#endif

#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/ComponentNode.h>
#include <Dataflow/Network/PackageDBHandler.h>
#include <Core/XMLUtil/StrX.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/Util/Environment.h>
#include <Core/Util/soloader.h>
#include <Core/Util/FileUtils.h>
#include <Core/OS/Dir.h> // for LSTAT

#include <stdio.h>
#include <iostream>
#include <ctype.h>
#include <string>
#include <vector>

using namespace std;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  define IRIX
#  pragma set woff 1375
#endif

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/sax/ErrorHandler.hpp>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  pragma reset woff 1375
#endif

#include <sys/stat.h>

#ifdef __APPLE__
  static string lib_ext = ".dylib";
#elif defined(_WIN32)
  const string lib_ext = ".dll";
#else
  static string lib_ext = ".so";
#endif

namespace SCIRun {

  PackageDB* packageDB = 0;

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
  : delayed_commands_(),
    db_(new Packages), 
    packageList_(0), 
    gui_(gui)
{
}

PackageDB::~PackageDB()
{ 
  delete db_; 
}

typedef void (*pkgInitter)(const string& tclPath);

bool
PackageDB::findMaker(ModuleInfo* moduleInfo)
{
  string cat_bname, pak_bname;
  if(moduleInfo->packageName == "SCIRun") {
    cat_bname = "Dataflow_Modules_";
    pak_bname = "Dataflow";
  } else {
    cat_bname = "Packages_" + moduleInfo->packageName + "_Dataflow_Modules_";
    pak_bname = "Packages_" + moduleInfo->packageName + "_Dataflow";
  }
  string errstr;

  // try the large version of the shared library
  LIBRARY_HANDLE package_so = findLib("lib" + pak_bname+lib_ext);
  if (!package_so)
    errstr = string(" - ")+SOError()+string("\n");

  // If package is FieldsChoose, FieldsCreate, FieldsHandle Fields Packages,
  // or UnuA-M, UnuN-Z
  string cat_name = moduleInfo->categoryName;
  if (cat_name.substr(0, 6) == "Fields") { cat_name = "Fields"; }
  else if (cat_name.substr(0, 7) == "UnuAtoM") { cat_name = "Unu"; }
  else if (cat_name.substr(0, 7) == "UnuNtoZ") { cat_name = "Unu"; }

  // try the small version of the shared library
  LIBRARY_HANDLE category_so = findLib("lib" + cat_bname+cat_name+lib_ext);
  if (!category_so)
    errstr = string(" - ")+SOError()+string("\n");


  if (!category_so && !package_so) {
    printMessage("Unable to load all of package '" + moduleInfo->packageName +
		 "' (category '" + moduleInfo->categoryName + "' failed) :\n" 
		 + errstr);
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
    // the messages happen elsewere...
    if (moduleInfo->optional != "true") {
      printMessage("Unable to load module '" + moduleInfo->moduleName +
		   "' :\n - can't find symbol '" + makename + "'\n");
    }
    return false;
  }
  return true;
}


void
PackageDB::loadPackage(bool resolve)
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

  printMessage("Loading packages, please wait...");

  //#ifdef __APPLE__
  //  // A hack around a gcc (apple) init bug
  //  GetLibrarySymbolAddress( "lib/libCore_Datatypes.dylib",
  //			   "__ZN6SCIRun10CurveFieldINS_6TensorEE2ioERNS_9PiostreamE");
  //#endif

  // the format of PACKAGE_PATH is a colon seperated list of paths to the
  // root(s) of package source trees.
  const char *srcdir = sci_getenv("SCIRUN_SRCDIR");
  ASSERT(srcdir);
  packagePath = srcdir + string("/Packages");

  // if the user specififes it, build the complete package path
  const char *packpath = sci_getenv("PACKAGE_SRC_PATH");
  if (packpath) packagePath = string(packpath) + ":" + packagePath;

  // the format of LOAD_PACKAGE is a comma seperated list of package names.
  // build the complete list of packages to load
  ASSERT(sci_getenv("SCIRUN_LOAD_PACKAGE"));
  loadPackage = string(sci_getenv("SCIRUN_LOAD_PACKAGE"));

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
#ifdef _WIN32
      // don't find the drive letter name's ':'...
      const unsigned int firstColon = tmpPath.find(':',2);
#else
      const unsigned int firstColon = tmpPath.find(':');
#endif
      if(firstColon < tmpPath.size()) {
	pathElt=tmpPath.substr(0,firstColon);
	tmpPath=tmpPath.substr(firstColon+1);
      } else {
	pathElt=tmpPath;
	tmpPath="";
      }
      
      struct stat buf;
      LSTAT((pathElt+"/"+packageElt).c_str(),&buf);
      if (S_ISDIR(buf.st_mode)) {
	tmpPath = "found";
	break;
      }
    }

    if (tmpPath=="") {
      printMessage("Unable to load package " + packageElt +
		   ":\n - Can't find " + packageElt +
		   " directory in package path");
      continue;
    }

    gui_exec("lappend auto_path "+pathElt+"/"+packageElt+"/Dataflow/GUI");

    string xmldir;
    
    if(packageElt == "SCIRun") {
      xmldir = string(srcdir) + "/Dataflow/XML";
      gui_exec("lappend auto_path "+string(srcdir)+"/Dataflow/GUI");
    } else {
      xmldir = pathElt+"/"+packageElt+"/Dataflow/XML";
      gui_exec(string("lappend auto_path ")+pathElt+"/"+packageElt+
		 "/Dataflow/GUI");
    }
    std::map<int,char*>* files;
    files = GetFilenamesEndingWith((char*)xmldir.c_str(),".xml");

    if (!files) {
      printMessage("Unable to load package " + packageElt +
		   ":\n - Couldn't find *.xml in " + xmldir );
      continue;
    }

    new_package = new package;
    new_package->name = packageElt;
    packages.insert(std::pair<int,
		    package*>(packages.size(),new_package));

    mod_count += files->size();

    for (char_iter i=files->begin();
	 i!=files->end();
	 i++) {
      if (node) DestroyComponentNode(node);
      node = CreateComponentNode(1);
      ReadComponentNodeFromFile(node,(xmldir+"/"+(*i).second).c_str(), gui_);

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
	new_module->optional = node->optional;
	new_module->packageName = packageElt;
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

  gui_exec("addProgressSteps " + to_string(mod_count));

  int index = 0;
  int numreg;
  
  for (pi = packages.begin();
       pi!=packages.end();
       pi++) {
    
    numreg = 0;
    
    string pname = (*pi).second->name;

    printMessage("Loading package '" + pname + "'");
    gui_exec("setProgressText {Loading package: " + pname + " }");

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
	    if (((*mi).second)->optional != "true") {
	      printMessage("Unable to load module '" + mname +
			   "' :\n - can't find symbol 'make_" + mname + "'");
	    }
	  }
	} else {
	  numreg++;
	  registerModule((*mi).second);
	}
	gui_exec("incrProgress");
      }
    }
    
    if (numreg) {
	gui_exec("createPackageMenu " + to_string(index++));
    } else {
      printMessage("Unable to load package " + pname + ":\n"
		   " - could not find any valid modules.");
    }
  }

  printMessage("\nFinished loading packages.");
}
  
void
PackageDB::registerModule(ModuleInfo* info) 
{
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
 
Module*
PackageDB::instantiateModule(const string& packageName,
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

  ASSERT(gui_);
  GuiContext* module_context = gui_->createContext(instanceName);
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
 
bool
PackageDB::haveModule(const string& packageName,
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

  // Look up the package name somewhere else.  Find a remapping.
  PackageIter iter(package);
  for (iter.first(); iter.ok();++iter)
  {
    if (iter.get_data()->lookup(modName, modinfo))
    {
      return iter.get_key();
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

void
PackageDB::setGui(GuiInterface* gui)
{
  gui_ = gui;
  if (gui_) {
    for(vector<string>::iterator iter = delayed_commands_.begin();
	iter != delayed_commands_.end(); ++iter){
      gui_->execute(*iter);
    }
    delayed_commands_.clear();
  }
}

void
PackageDB::gui_exec(const string& command)
{
  if(gui_)
    gui_->execute(command);
  else
    delayed_commands_.push_back(command);
}

ModuleInfo*
PackageDB::GetModuleInfo(const string& name,
			 const string& catname,
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


void
PackageDB::printMessage(const string &msg) 
{
  if(gui_){
    gui_->postMessage(msg);
    gui_->execute("update idletasks");
  } else {
    cerr << msg << "\n";
  }
}

  
