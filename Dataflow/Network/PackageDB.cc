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

#include <Core/Util/soloader.h>
#ifdef ASSERT
#undef ASSERT
#endif
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/FileUtils.h>
#include <Dataflow/Network/ComponentNode.h>
#include <Dataflow/Network/PackageDBHandler.h>
#include <Dataflow/Network/StrX.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <stdio.h>
#include <iostream>
#include <ctype.h>
using std::cerr;
using std::ostream;
using std::endl;
using std::cout;
#include <vector>
using std::vector;

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include <sax/ErrorHandler.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

namespace SCIRun {

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


PackageDB packageDB;

PackageDB::PackageDB(void) :
    db_((void*)new Packages), packageList_(0)
{
}

PackageDB::~PackageDB(void)
{ 
  delete (Packages*)db_; 
}

typedef void (*pkgInitter)(const string& tclPath);

void PackageDB::loadPackage(const string& packPath)
{
  string packagePath = packPath;
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

  postMessage("Loading packages, please wait...\n", false);


  // The format of a package path element is either URL,URL,...
  // Where URL is a filename or a url to an XML file that
  // describes the components in the package.
  
  while(packagePath!="") {
    // Strip off the first element, leave the rest in the path for the next
    // iteration.
    const unsigned int firstComma = packagePath.find(',');
    if(firstComma < packagePath.size()) {
      packageElt=packagePath.substr(0,firstComma);
      packagePath=packagePath.substr(firstComma+1,-1);
    } else {
      packageElt=packagePath;
      packagePath="";
    }

    TCL::execute(string("lappend auto_path ")+packageElt+"/Dataflow/GUI");
    
    string bname = basename(packageElt);
    string pname = basename(packageElt);
    
    if(bname == "src") {
      bname = "";
      pname = "SCIRun";
    } else {
      bname = "Packages_" + bname + "_";
    }

    string xmldir = packageElt+"/Dataflow/XML";
    std::map<int,char*>* files;
    files = GetFilenamesEndingWith((char*)xmldir.c_str(),".xml");

    if (!files) {
      postMessage("Unable to load package " + pname +
		  ":\n - Couldn't find " + xmldir + " directory");
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
      node = CreateComponentNode(3);
      ReadComponentNodeFromFile(node,(xmldir+"/"+(*i).second).c_str());

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

  TCL::execute("toplevel .loading; "
	       "wm geometry .loading 250x75+275+200; "
	       "wm title .loading {Loading packages}; "
	       "update idletasks");
  TCL::execute("iwidgets::feedback .loading.fb -labeltext "
	       "{Loading package:                 }"
	       " -steps " + to_string(mod_count) + ";"
	       "pack .loading.fb -padx 5 -fill x; update idletasks");

  LIBRARY_HANDLE package_so;
  LIBRARY_HANDLE category_so;
  string libname;
  string cat_bname,pak_bname;
  string pname,cname,mname;
  string category_error;
  string package_error;
  string makename;
  string command;
  int index = 0;
  int numreg;
  
  mod_count = 0;

  for (pi = packages.begin();
       pi!=packages.end();
       pi++) {

    numreg = 0;
    
    pname = (*pi).second->name;

    if(pname == "SCIRun") {
      cat_bname = "Dataflow_Modules_";
      pak_bname = "Dataflow";
    } else {
      cat_bname = "Packages_" + pname + "_Dataflow_Modules_";
      pak_bname = "Packages_" + pname + "_Dataflow";
    }

    postMessage("Loading package '" + pname + "'", false);
    TCL::execute(".loading.fb configure -labeltext {Loading package: " +
		 pname + " }");
    TCL::eval("update idletasks",result);

    // try the large version of the .so
    libname = "lib" + pak_bname + ".so";
    package_so = GetLibraryHandle(libname.c_str());
    if (!package_so)
      package_error = SOError();

    for (ci = (*pi).second->categories.begin();
	 ci!=(*pi).second->categories.end();
	 ci++) {

      cname = (*ci).second->name;

      // try the small version of the .so 
      libname = "lib" + cat_bname + cname + ".so";
      category_so = GetLibraryHandle(libname.c_str());
      if (!category_so)
	category_error = SOError();

      if (!category_so && !package_so) {
	postMessage("Unable to load all of package '" + pname +
		    "' (category '" + cname + "' failed) :\n - " +
		    package_error + "\n - " + category_error + "\n");
	TCL::execute("update idletasks");
	continue;
      }

      for (mi = (*ci).second->modules.begin();
	   mi!=(*ci).second->modules.end();
	   mi++) {
	mname = (*mi).second->moduleName;
	makename = "make_" + mname;
	(*mi).second->maker = 0;
	if (category_so)
	  (*mi).second->maker = 
	    (ModuleMaker)GetHandleSymbolAddress(category_so,makename.c_str());
	if (!(*mi).second->maker && package_so)
	  (*mi).second->maker = 
	    (ModuleMaker)GetHandleSymbolAddress(package_so,makename.c_str());
	if (!(*mi).second->maker) {
	  postMessage("Unable to load module '" + mname +
		      "' :\n - can't find symbol 'make_" + mname + "'\n");
	  TCL::execute("update idletasks");
	  //destroy new_module here
	  continue;
	} else {
	  numreg++;
	  registerModule((*mi).second);
	}

	TCL::execute(string("if [winfo exists .loading.fb] ") + 
		     string("{.loading.fb step; update idletasks}"));
      }
    }
    
    if (numreg) {
      command = "createPackageMenu " + to_string(index++);
      TCL::execute(command);
      TCL::execute("update idletasks");
    } else 
      postMessage("Unable to load package " + pname + ":\n"
                  " - could not find any valid modules.\n");
  }

  postMessage("\nFinished loading packages.\n",false);
  TCL::execute(string("if [winfo exists .loading] {destroy .loading}"));
  TCL::eval("update idletasks",result);
}
  
void PackageDB::registerModule(ModuleInfo* info) {
  Packages* db=(Packages*)db_;
 
  Package* package;
  if(!db->lookup(info->packageName,package))
    {
      db->insert(info->packageName,package=new Package);
      packageList_.add( info->packageName );
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
  Packages* db=(Packages*)db_;
  
  Package* package;
  if(!db->lookup(fromPackageName,package)) {
    postMessage("Warning: creating an alias from a nonexistant package "+fromPackageName+" (ignored)");
    return;
  }
  
  Category* category;
  if(!package->lookup(fromCategoryName,category)) {
    postMessage("Warning: creating an alias from a nonexistant category "+fromPackageName+"."+fromCategoryName+" (ignored)");
    return;
  }
  
  ModuleInfo* moduleInfo;
  if(!category->lookup(fromModuleName,moduleInfo)) {
    postMessage("Warning: creating an alias from a nonexistant module "+fromPackageName+"."+fromCategoryName+"."+fromModuleName+" (ignored)");
    return;
  }
  registerModule(moduleInfo);
}
 
Module* PackageDB::instantiateModule(const string& packageName,
				     const string& categoryName,
				     const string& moduleName,
				     const string& instanceName) const {
  Packages* db=(Packages*)db_;

  Package* package;
  if(!db->lookup(packageName,package)) {
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
  
  Module *module = (moduleInfo->maker)(instanceName);
  
  // Some modules may already know their package and category.
  // If this module doesn't, then set it's package and category here.
  string unknown("unknown");
  if (unknown == module->packageName)
    module->packageName=packageName;
  if (unknown == module->categoryName)
    module->categoryName=categoryName;

  // copy other fields 
  module->lastportdynamic = moduleInfo->lastportdynamic;
  
  return module;
}
 
Array1<string> PackageDB::packageNames(void) const {
   
  // packageList_ is used to keep a list of the packages 
  // that are in this PSE IN THE ORDER THAT THEY ARE SPECIFIED
  // by the user in the Makefile (for main.cc) or in their
  // environment.
  
  return packageList_;
}

Array1<string>
PackageDB::categoryNames(const string& packageName) const {
  Packages* db=(Packages*)db_;
  {
    PackagesIter iter(db);
    for(iter.first();iter.ok();++iter) if(iter.get_key()==packageName) {
      Package* package=iter.get_data();
      Array1<string> result(package->size());
      {
	PackageIter iter(package);
	int i=0;
	for(iter.first();iter.ok();++iter) result[i++]=iter.get_key();
      }
      return result;
    }
  }
  cerr << "WARNING: Unknown package " << packageName << "\n";
  
  Array1<string> result(0);
  return result;
}
 
Array1<string>
PackageDB::moduleNames(const string& packageName,
		       const string& categoryName) const {
  Packages* db=(Packages*)db_;
  {
    PackagesIter iter(db);
    for(iter.first();iter.ok();++iter) 
      if(iter.get_key()==packageName) {
	Package* package=iter.get_data();
	{
	  PackageIter iter(package);
	  for(iter.first();iter.ok();++iter) 
	    if(iter.get_key()==categoryName) {
	      Category* category=iter.get_data();
	      Array1<string> result(category->size());
	      {
		CategoryIter iter(category);
		int i=0;
		for(iter.first();iter.ok();++iter) 
		  result[i++]=iter.get_key();
	      }
	      return result;
	    }
	  cerr << "WARNING: Unknown category " << packageName << "."
	       << categoryName << "\n";
	}
      }
  }
 cerr << "WARNING: Unknown package " << packageName << "\n";
 
 Array1<string> result(0);
 return result;
}

} // End namespace SCIRun
