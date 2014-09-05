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

// PackageDB.h - Interface to module-finding and loading mechanisms

#ifndef PSE_Dataflow_PackageDB_h
#define PSE_Dataflow_PackageDB_h 1

#include <Dataflow/share/share.h>

#include <Core/Containers/Array1.h>
#include <Core/Containers/AVLTree.h>
#include <Core/Containers/String.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {


    typedef struct {
      clString name;
      clString datatype;
      iport_maker maker;
    } IPortInfo;

    typedef struct {
      clString name;
      clString datatype;
      oport_maker maker;
    } OPortInfo;

    typedef struct {
      clString packageName;
      clString categoryName;
      clString moduleName;
      ModuleMaker maker;
      clString uiFile;
      std::map<int,IPortInfo*>* iports;
      std::map<int,OPortInfo*>* oports;
      char lastportdynamic;
    } ModuleInfo;

    typedef AVLTree<clString,ModuleInfo*> Category;
    typedef AVLTree<clString,Category*> Package;
    typedef AVLTree<clString,Package*> Packages;
    
    typedef AVLTreeIter<clString,ModuleInfo*> CategoryIter;
    typedef AVLTreeIter<clString,Category*> PackageIter;
    typedef AVLTreeIter<clString,Package*> PackagesIter;

    class PSECORESHARE PackageDB {
      public:
        PackageDB(void);
        ~PackageDB(void);

        void loadPackage(const clString& packagePath);
        void registerModule(ModuleInfo* info);
	void createAlias(const clString& fromPackageName,
			 const clString& fromCategoryName,
			 const clString& fromModuleName,
			 const clString& toPackageName,
			 const clString& toCategoryName,
			 const clString& toModuleName);

        Module* instantiateModule(const clString& packageName,
                                  const clString& categoryName,
                                  const clString& moduleName,
                                  const clString& instanceName) const;

        Array1<clString> packageNames(void) const;
        Array1<clString> categoryNames(const clString& packageName) const;
        Array1<clString> moduleNames(const clString& packageName,
                                     const clString& categoryName) const;
      public:
        void *             db_;
        Array1<clString>   packageList_;
    };

    // PackageDB is intended to be a singleton class, but nothing will break
    // if you instantiate it many times.  This is the singleton instance,
    // on which you should invoke operations:

    PSECORESHARE extern PackageDB packageDB;

} // End namespace SCIRun

#endif // PSE_Dataflow_PackageDB_h
