// PackageDB.h - Interface to module-finding and loading mechanisms

#ifndef PSE_Dataflow_PackageDB_h
#define PSE_Dataflow_PackageDB_h 1

#include <PSECore/share/share.h>

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/AVLTree.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>

namespace PSECore {
  namespace Dataflow {

    using SCICore::Containers::clString;
    using SCICore::Containers::Array1;
    using SCICore::Containers::AVLTree;
    using SCICore::Containers::AVLTreeIter;

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
        void *             d_db;
        Array1<clString>   d_packageList;
    };

    // PackageDB is intended to be a singleton class, but nothing will break
    // if you instantiate it many times.  This is the singleton instance,
    // on which you should invoke operations:

    PSECORESHARE extern PackageDB packageDB;

  }
}

#endif // PSE_Dataflow_PackageDB_h
