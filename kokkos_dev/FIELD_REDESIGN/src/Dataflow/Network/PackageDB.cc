// PackageDB.cc - Interface to module-finding and loading mechanisms
// $Id$

#include <SCICore/Util/soloader.h>
#ifdef ASSERT
#undef ASSERT
#endif
#include <SCICore/Containers/AVLTree.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/PackageDB.h>
#include <PSECore/Dataflow/FileUtils.h>
#include <PSECore/Dataflow/ComponentNode.h>
#include <PSECore/Dataflow/PackageDBHandler.h>
#include <PSECore/Dataflow/StrX.h>
#include <PSECore/Dataflow/NetworkEditor.h>
#include <PSECore/XMLUtil/XMLUtil.h>
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

namespace PSECore {
namespace Dataflow {

using namespace SCICore::Containers;
using SCICore::Containers::AVLTree;
using SCICore::Containers::AVLTreeIter;
using namespace PSECore::XMLUtil;

typedef struct ModuleInfo_tag {
  ModuleMaker maker;
  clString uiFile;
} ModuleInfo;

typedef AVLTree<clString,ModuleInfo*> Category;
typedef AVLTree<clString,Category*> Package;
typedef AVLTree<clString,Package*> Packages;

typedef AVLTreeIter<clString,ModuleInfo*> CategoryIter;
typedef AVLTreeIter<clString,Category*> PackageIter;
typedef AVLTreeIter<clString,Package*> PackagesIter;

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
  while(packagePath!="") {
    
    // Strip off the first element, leave the rest in the path for the next
    // iteration.
    
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

    // The GUI path is hard-wired to be "PACKAGENAME/GUI""
    TCL::execute(clString("lappend auto_path ")+packageElt+"/GUI");

    // get *.xml in the above GUI directory.
    clString xmldir = packageElt+"/XML";
    std::map<int,char*>* files = 
      GetFilenamesEndingWith((char*)xmldir(),".xml");
    component_node* node = 0;
    for (char_iter i=files->begin();
	 i!=files->end();
	 i++) {
      if (node) DestroyComponentNode(node);
      node = CreateComponentNode(3);
      ReadComponentNodeFromFile(node,(packageElt+"/XML/"+(*i).second)());
      
      // find the .so for this component
      LIBRARY_HANDLE so;
      ModuleMaker makeaddr = 0;
      clString libname(clString("lib")+basename(packageElt)+"_Modules_"+
		       node->category+".so");
      so = GetLibraryHandle(libname());
      if (!so) {
	clString firsterror(SOError());
	libname = clString("lib")+basename(packageElt)+".so";
	so = GetLibraryHandle(libname());
	if (!so)
	  postMessage("PackageDB: Couldn't load all of package \\\""+
		      basename(packageElt)+"\\\"\n  "+
		      firsterror()+"\n  "+SOError());
      }
      
      if (so) {
	clString make(clString("make_")+node->name);
	makeaddr = (ModuleMaker)GetHandleSymbolAddress(so,make());
	if (!makeaddr)
	  postMessage(clString("PackageDB: Couldn't find component \\\"")+
		      node->name+"\\\"\n  "+
		      SOError());
      }
      
      if (makeaddr)
	registerModule(basename(packageElt()),node->category,
		       node->name,makeaddr,"not currently used");
    }
  }
  
  TCL::execute("createCategoryMenu");
}
  
void PackageDB::registerModule(const clString& packageName,
			       const clString& categoryName,
			       const clString& moduleName,
			       ModuleMaker moduleMaker,
			       const clString& tclUIFile) {
  Packages* db=(Packages*)d_db;
 
  Package* package;
  if(!db->lookup(packageName,package))
    {
      db->insert(packageName,package=new Package);
      d_packageList.add( packageName );
    }
  
  Category* category;
  if(!package->lookup(categoryName,category))
    package->insert(categoryName,category=new Category);
  
  ModuleInfo* moduleInfo;
  if(!category->lookup(moduleName,moduleInfo)) {
    moduleInfo=new ModuleInfo;
    category->insert(moduleName,moduleInfo);
  } else cerr << "WARNING: Overriding multiply registered module "
	      << packageName << "." << categoryName << "."
	      << moduleName << "\n";
  
  moduleInfo->maker=moduleMaker;
  moduleInfo->uiFile=tclUIFile;
}
 
void PackageDB::createAlias(const clString& fromPackageName,
			    const clString& fromCategoryName,
			    const clString& fromModuleName,
			    const clString& toPackageName,
			    const clString& toCategoryName,
			    const clString& toModuleName)
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
  registerModule(toPackageName, toCategoryName, toModuleName,
		 moduleInfo->maker, moduleInfo->uiFile);
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
  //
  // Steve understandably doesn't like new, fragile mechanisms where
  // perfectly good old, traditional ones already exist, so he if0'd this
  // away and added the "lappend auto_path" at package-load-time, above.
  //
  // This code is still here 'cause Some Day it might be nice to allow the
  // source of the TCL files to be stored in the .so (as strings) and eval'd
  // here.  This was the "faraway vision" that drove me to do things this way
  // in the first place, but since that vision seems to have stalled
  // indefinately in lieu of Useful Work, there's no reason not to use
  // auto_path (except that it produces yet one more file to maintain).  And
  // auto_path is useful if you write global f'ns and want to use them in lots
  // of your modules -- auto_path nicely handles this whereas the code below
  // doesn't handle it at all.
  //
  // Some day it might be nice to actually achieve the "package is one .so
  // and that's all" vision, but not today.  :)
  //
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
  module->packageName = packageName;
  module->moduleName = moduleName;
  module->categoryName = categoryName;
  
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

} // Dataflow namespace
} // PSECore namespace

//
// $Log$
// Revision 1.15.2.2  2000/10/26 14:16:50  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.23  2000/10/24 05:57:41  moulding
// new module maker Phase 2: new module maker goes online
//
// These changes clean out the last remnants of the old module maker and
// bring the new module maker online.
//
// Revision 1.22  2000/10/22 21:27:09  moulding
// cleaned up code associated with the new module maker
//
// Revision 1.21  2000/10/21 18:46:06  moulding
// turned new module maker off ... again.
//
// Revision 1.20  2000/10/21 18:33:44  moulding
// removed the PackageDBHandler and StrX classes from PackageDB.cc and put them
// into their own files.  This allows other pieces of code to use those classes.
//
// Revision 1.19  2000/10/19 15:13:26  moulding
// set NEW_MODULE_MAKER to 0;  I accidentally committed a version with it set
// to 1.
//
// Revision 1.18  2000/10/19 08:05:02  moulding
// cordoned off an old section of code with #if !NEW_MODULE_MAKER that
// traversed the old components.xml file.  It will be defunct soon, and will
// be removed at that time.
//
// Revision 1.17  2000/10/19 07:58:49  moulding
// - finishing touches for phase 1 of new module maker.
//
// - added more useful messages when package loading fails.
//
// Revision 1.16  2000/10/18 17:37:00  moulding
// added sections of code cordoned off with #if NEW_MODULE_MAKER which will
// help facilitate the move to the new module maker and the move to source forge.
//
// Revision 1.15  2000/03/17 08:24:52  sparker
// Added XML parser for component repository
//
// Revision 1.14  1999/10/07 02:07:20  sparker
// use standard iostreams and complex type
//
// Revision 1.13  1999/10/06 20:37:37  mcq
// Added memoirs.
//
// Revision 1.12  1999/09/22 23:51:46  dav
// removed debug print
//
// Revision 1.11  1999/09/22 22:39:50  dav
// updated to use tclIndex files
//
// Revision 1.10  1999/09/08 02:26:41  sparker
// Various #include cleanups
//
// Revision 1.9  1999/09/04 06:01:41  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.8  1999/08/31 23:27:53  sparker
// Added Log and Id entries
//
//
