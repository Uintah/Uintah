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

#include <sci_defs.h>

#include <Dataflow/share/share.h>

#include <Core/Containers/AVLTree.h>
#include <Core/Util/soloader.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {
  class GuiInterface;
  using namespace std;
    typedef struct {
      string name;
      string datatype;
      iport_maker maker;
    } IPortInfo;

    typedef struct {
      string name;
      string datatype;
      oport_maker maker;
    } OPortInfo;

    typedef struct {
      string packageName;
      string categoryName;
      string moduleName;
      string help_description;
      ModuleMaker maker;
      string uiFile;
      std::map<int,IPortInfo*>* iports;
      std::map<int,OPortInfo*>* oports;
      char lastportdynamic;
    } ModuleInfo;

    typedef AVLTree<string,ModuleInfo*> Category;
    typedef AVLTree<string,Category*> Package;
    typedef AVLTree<string,Package*> Packages;
    
    typedef AVLTreeIter<string,ModuleInfo*> CategoryIter;
    typedef AVLTreeIter<string,Category*> PackageIter;
    typedef AVLTreeIter<string,Package*> PackagesIter;

    class PSECORESHARE PackageDB {
    public:
      PackageDB(GuiInterface* gui);
      ~PackageDB();

      void loadPackage(bool resolve=true);
      void registerModule(ModuleInfo* info);
      void createAlias(const string& fromPackageName,
		       const string& fromCategoryName,
		       const string& fromModuleName,
		       const string& toPackageName,
		       const string& toCategoryName,
		       const string& toModuleName);

      Module* instantiateModule(const string& packageName,
				const string& categoryName,
				const string& moduleName,
				const string& instanceName);

      bool haveModule(const string& packageName,
		      const string& categoryName,
		      const string& moduleName) const;

      vector<string> packageNames() const;
      vector<string> categoryNames(const string& packageName) const;
      vector<string> moduleNames(const string& packageName,
                                     const string& categoryName) const;
      void setGui(GuiInterface* gui);
      ModuleInfo* GetModuleInfo(const string& name, const string& catname,
				const string& packname);

      // Used if the module has changed categories.
      string getCategoryName(const string &packName,
			     const string &catName,
			     const string &modName);
      void setSplashPath(string p);
    private:
      LIBRARY_HANDLE findLibInPath(string,string);
      bool findMaker(ModuleInfo* info);

      vector<string> delayed_commands;
      void do_command(const string& cmd);
      Packages *             db_;
      vector<string>     packageList_;
      GuiInterface* gui;
      string splash_path_;
    };

    // PackageDB is intended to be a singleton class, but nothing will break
    // if you instantiate it many times.  This is the singleton instance,
    // on which you should invoke operations:

    PSECORESHARE extern PackageDB* packageDB;

} // End namespace SCIRun

#endif // PSE_Dataflow_PackageDB_h
