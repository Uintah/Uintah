// PackageDB.cc - Interface to module-finding and loading mechanisms
// $Id$


#include <SCICore/Util/soloader.h>
#ifdef ASSERT
#undef ASSERT
#endif
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Containers/AVLTree.h>
#include <SCICore/Containers/String.h>
#include "PackageDB.h"

namespace PSECore {
namespace Dataflow {

using namespace SCICore::Containers;
using SCICore::Containers::AVLTree;
using SCICore::Containers::AVLTreeIter;

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

PackageDB::PackageDB(void): _db((void*)new Packages) { }
PackageDB::~PackageDB(void) { delete (Packages*)_db; }

typedef void (*pkgInitter)(const clString& tclPath);

void PackageDB::loadPackage(const clString& packPath) {

  // The format of a package path element is either "soLib" or "soLib(tclPath)"
  // where soLib is the name of a package .so file, and tclPath is the path
  // to use to locate that package's tcl files.  If the first form is used,
  // the tclPath is constructed by substituting "TCL" for everything in the
  // soLib string after the final "/".
  //
  // A package path is a colon-separated list of package path elements.

  clString packagePath(packPath);     // Copy to deal with non-const methods
  while(packagePath!="") {

    // Strip off the first element, leave the rest in the path for the next
    // iteration.

    clString packageElt;
    int firstColon=packagePath.index(':');
    if(firstColon!=-1) {
      packageElt=packagePath.substr(0,firstColon);
cerr << "Before '" << packagePath << "'\n";
      packagePath=packagePath.substr(firstColon+1,-1);
cerr << "After '" << packagePath << "'\n";
    } else {
      packageElt=packagePath;
      packagePath="";
    }

    // Parse the element apart into soName and tclPath, using the default
    // tclpath (soName's directory + "/GUI") if there isn't one specified.

    clString soName;
    clString tclPath;

    int openParen=packageElt.index('(');
    if(openParen!=-1) {
      int closeParen=packageElt.index(')');
      soName=packageElt.substr(0,openParen);
      tclPath=packageElt.substr(openParen+1,closeParen-openParen-1);
    } else {
      soName=packageElt;
      if(pathname(packageElt)!="")
        tclPath=pathname(packageElt)+"/GUI";
      else
        tclPath=pathname(packageElt)+"GUI";
    }

    // Load the package

    {
      clString result;
      TCL::eval(clString(".top.errorFrame.text insert end \"Loading package '")
                +soName+"' with TCLPath '"+tclPath+"'\\n\"",result);
    }

    //void* so=dlopen(soName(),RTLD_NOW);
	LIBRARY_HANDLE so = GetLibraryHandle(soName());
    if(!so) {
	  //cerr << dlerror() << '\n';
      cerr << "ERROR: Can't open package '" << soName << "'\n";
      continue;
    }
    //pkgInitter initFn=(pkgInitter)dlsym(so,"initPackage");
	pkgInitter initFn=(pkgInitter)GetHandleSymbolAddress(so,"initPackage");
    if(!initFn) {
      cerr << "ERROR: Package '" << soName << "' has no initPackage(...)\n";
      continue;
    }
    initFn(tclPath);

    // You can't close the sofile; it loads more stuff in when you instantiate
    // a module -- all the linking does not occur here.

    // XXX: We need to keep the handle around to avoid opening the same one
    //      a zillion times, and to close them off when you're in a development
    //      cycle so the old inodes can get freed.

  }

  clString result;
  TCL::eval("createCategoryMenu",result);
}

void PackageDB::registerModule(const clString& packageName,
                               const clString& categoryName,
                               const clString& moduleName,
                               ModuleMaker moduleMaker,
                               const clString& tclUIFile) {
  Packages* db=(Packages*)_db;

  Package* package;
  if(!db->lookup(packageName,package))
    db->insert(packageName,package=new Package);

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

Module* PackageDB::instantiateModule(const clString& packageName,
                                     const clString& categoryName,
                                     const clString& moduleName,
                                     const clString& instanceName) const {
  Packages* db=(Packages*)_db;

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

  // Source the UI file if there is one and we haven't yet

  if(moduleInfo->uiFile!="") {
    clString result;
    if(!TCL::eval(clString("source ")+moduleInfo->uiFile,result)) {
      cerr << "Can't source UI file " << moduleInfo->uiFile << "...\n";
      cerr << "  TCL Error: " << result << "\n";
    }
    moduleInfo->uiFile="";                       // Don't do it again
  }

  Module *module = (moduleInfo->maker)(instanceName);
  module->packageName = packageName;
  module->moduleName = moduleName;
  module->categoryName = categoryName;

  return module;
}

Array1<clString> PackageDB::packageNames(void) const {
  Packages* db=(Packages*)_db;

  Array1<clString> result(db->size());
  {
    PackagesIter iter(db);
    int i=0;
    for(iter.first();iter.ok();++iter) result[i++]=iter.get_key();
  }
  return result;
}

Array1<clString> PackageDB::categoryNames(const clString& packageName) const {
  Packages* db=(Packages*)_db;

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

Array1<clString> PackageDB::moduleNames(const clString& packageName,
                                        const clString& categoryName) const {
  Packages* db=(Packages*)_db;

  {
    PackagesIter iter(db);
    for(iter.first();iter.ok();++iter) if(iter.get_key()==packageName) {
      Package* package=iter.get_data();
      {
        PackageIter iter(package);
        for(iter.first();iter.ok();++iter) if(iter.get_key()==categoryName) {
          Category* category=iter.get_data();
          Array1<clString> result(category->size());
          {
            CategoryIter iter(category);
            int i=0;
            for(iter.first();iter.ok();++iter) result[i++]=iter.get_key();
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
// Revision 1.8  1999/08/31 23:27:53  sparker
// Added Log and Id entries
//
//
