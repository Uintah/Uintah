// PackageDB.cc - Interface to module-finding and loading mechanisms

#include <Core/Util/soloader.h>
#ifdef ASSERT
#undef ASSERT
#endif
#include <Core/Containers/String.h>
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

typedef std::map<int,char*>::iterator char_iter;
typedef std::map<int,inport_node*>::iterator inport_iter;
typedef std::map<int,outport_node*>::iterator outport_iter;

namespace SCIRun {


PackageDB packageDB;

PackageDB::PackageDB(void) :
    d_db((void*)new Packages), d_packageList(0)
{
}

PackageDB::~PackageDB(void)
{ 
  delete (Packages*)d_db; 
}

typedef void (*pkgInitter)(const clString& tclPath);

void PackageDB::loadPackage(const clString& packPath)
{
  // The format of a package path element is either URL,URL,...
  // Where URL is a filename or a url to an XML file that
  // describes the components in the package.
  clString packagePath = packPath;
  clString result;
  char string[100];
  int index=0;

  bool loading = false;

  postMessage("Loading packages, please wait...\n", false);
  postMessage(clString("DEBUG: package path is: '")+packagePath, false);
  while(packagePath!="") {
    // Strip off the first element, leave the rest in the path for the next
    // iteration.
    postMessage(packagePath, false);
    clString packageElt;
    int firstComma=packagePath.index(',');
    if(firstComma!=-1) {
      packageElt=packagePath.substr(0,firstComma);
      packagePath=packagePath.substr(firstComma+1,-1);
    } else {
      packageElt=packagePath;
      packagePath="";
    }
    
    // Load the package
    postMessage(clString("Loading package '")+packageElt+"'", false);
    TCL::eval("update idletasks",result);

    // The GUI path is hard-wired to be "PACKAGENAME/GUI""
    TCL::execute(clString("lappend auto_path ")+packageElt+"/Dataflow/GUI");

    // get *.xml in the PACKAGENAME/XML directory.
    clString xmldir = packageElt+"/Dataflow/XML";
    std::map<int,char*>* files = 
      GetFilenamesEndingWith((char*)xmldir(),".xml");

    if ( !loading ) {
      TCL::execute(clString("toplevel .loading; "
			    "wm geometry .loading 250x75+275+200;"));
      loading = true;
    }
    TCL::execute(clString("iwidgets::feedback .loading.fb -labeltext "
			  + packageElt +
			  " -steps " + to_string(int(files->size())) + ";"
			  "pack .loading.fb; update idletasks"));
    component_node* node = 0;
    for (char_iter i=files->begin();
	 i!=files->end();
	 i++) {
      if (node) DestroyComponentNode(node);
      node = CreateComponentNode(3);
      ReadComponentNodeFromFile(node,(packageElt+"/Dataflow/XML/"+(*i).second)());
      
      // find the .so for this component
      LIBRARY_HANDLE so;
      ModuleMaker makeaddr = 0;
      
      clString bname = basename(packageElt);
      clString pname = basename(packageElt);

      if(bname == "src") {
	bname = "";
	pname = "SCIRun";
      } else {
	bname = clString("Packages_")+bname+clString("_");
      }

      clString libname(clString("lib")+bname+"Dataflow_Modules_"+
		       node->category+".so");
      so = GetLibraryHandle(libname());
      if (!so) {
	clString firsterror(SOError());
	libname = clString("lib")+pname+".so";
	so = GetLibraryHandle(libname());
	if (!so) {
	  postMessage("PackageDB: Couldn't load all of package \\\""+
		      pname+"\\\"\n  "+
		      firsterror()+"\n  "+SOError());
	  TCL::eval("update idletasks",result);
	}
      }
      
      if (so) {
	clString make(clString("make_")+node->name);
	makeaddr = (ModuleMaker)GetHandleSymbolAddress(so,make());
	if (!makeaddr) {
	  postMessage(clString("PackageDB: Couldn't find component \\\"")+
		      node->name+"\\\"\n  "+
		      SOError());
	  TCL::eval("update idletasks",result);
	}
      }
      
      if (makeaddr) {
	IPortInfo* ipinfo;
	OPortInfo* opinfo;
	ModuleInfo* info = scinew ModuleInfo;
	info->packageName = pname;
	info->categoryName = node->category;
	info->moduleName = node->name;
	info->maker = (ModuleMaker)makeaddr;
	info->uiFile = "not currently used";
	info->iports = scinew std::map<int,IPortInfo*>;
	info->oports = scinew std::map<int,OPortInfo*>;
	info->lastportdynamic = node->io->lastportdynamic;
	for (inport_iter i1 = node->io->inports->begin();
	     i1!=node->io->inports->end();
	     i1++) {
	  ipinfo = scinew IPortInfo;
	  ipinfo->name = clString(((*i1).second)->name);
	  ipinfo->datatype = clString(((*i1).second)->datatype);
	  ipinfo->maker = (iport_maker)0;
	  info->iports->insert(
	    std::pair<int,IPortInfo*>(info->iports->size(),
					    ipinfo));
	}
	for (outport_iter i2 = node->io->outports->begin();
	     i2!=node->io->outports->end();
	     i2++) {
	  opinfo = scinew OPortInfo;
	  opinfo->name = clString(((*i2).second)->name);
	  opinfo->datatype = clString(((*i2).second)->datatype);
	  opinfo->maker = (oport_maker)0;
	  info->oports->insert(
	    std::pair<int,OPortInfo*>(info->oports->size(),
					    opinfo));
	}
	registerModule(info);
	TCL::execute(clString(".loading.fb step"));
	//postMessageNoCRLF(".",false);
	TCL::eval("update idletasks",result);
      }
    }
    TCL::eval("update idletasks",result);
    sprintf(string,"createPackageMenu %d",index++);
    TCL::execute(string);
    TCL::execute(clString("destroy .loading.fb"));
  }
  
  postMessage("\nFinished loading packages.\n",false);
  TCL::execute(clString("destroy .loading"));
  TCL::eval("update idletasks",result);
  
  // don't do this.  Instead, create each package menu as it's loaded.
  //  TCL::execute("createCategoryMenu");
}
  
void PackageDB::registerModule(ModuleInfo* info) {
  Packages* db=(Packages*)d_db;
 
  Package* package;
  if(!db->lookup(info->packageName,package))
    {
      db->insert(info->packageName,package=new Package);
      d_packageList.add( info->packageName );
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
 
void PackageDB::createAlias(const clString& fromPackageName,
			    const clString& fromCategoryName,
			    const clString& fromModuleName,
			    const clString&,// toPackageName,
			    const clString&,// toCategoryName,
			    const clString&)// toModuleName)
{
  Packages* db=(Packages*)d_db;
  
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
 
Module* PackageDB::instantiateModule(const clString& packageName,
				     const clString& categoryName,
				     const clString& moduleName,
				     const clString& instanceName) const {
  Packages* db=(Packages*)d_db;

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
    clString result;
    if(!TCL::eval(clString("source ")+moduleInfo->uiFile,result)) {
      cerr << "Can't source UI file " << moduleInfo->uiFile << "...\n";
      cerr << "  TCL Error: " << result << "\n";
    }
    moduleInfo->uiFile="";                       // Don't do it again
  }
#endif
  
  Module *module = (moduleInfo->maker)(instanceName);
  
  // Some modules may already know their package and category.
  // If this module doesn't, then set it's package and category here.
  clString unknown("unknown");
  if (unknown == module->packageName)
    module->packageName=packageName;
  if (unknown == module->categoryName)
    module->categoryName=categoryName;

  // copy other fields 
  module->lastportdynamic = moduleInfo->lastportdynamic;
  
  return module;
}
 
Array1<clString> PackageDB::packageNames(void) const {
   
  // d_packageList is used to keep a list of the packages 
  // that are in this PSE IN THE ORDER THAT THEY ARE SPECIFIED
  // by the user in the Makefile (for main.cc) or in their
  // environment.
  
  return d_packageList;
}

Array1<clString>
PackageDB::categoryNames(const clString& packageName) const {
  Packages* db=(Packages*)d_db;
  {
    PackagesIter iter(db);
    for(iter.first();iter.ok();++iter) if(iter.get_key()==packageName) {
      Package* package=iter.get_data();
      Array1<clString> result(package->size());
      {
	PackageIter iter(package);
	int i=0;
	for(iter.first();iter.ok();++iter) result[i++]=iter.get_key();
      }
      return result;
    }
  }
  cerr << "WARNING: Unknown package " << packageName << "\n";
  
  Array1<clString> result(0);
  return result;
}
 
Array1<clString>
PackageDB::moduleNames(const clString& packageName,
		       const clString& categoryName) const {
  Packages* db=(Packages*)d_db;
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
	      Array1<clString> result(category->size());
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
 
 Array1<clString> result(0);
 return result;
}

} // End namespace SCIRun
