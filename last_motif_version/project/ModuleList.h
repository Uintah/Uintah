
/*
 *  ModuleList.h: Interface to Module database
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ModuleList_h
#define SCI_project_ModuleList_h 1

#include <Classlib/AVLTree.h>
#include <Classlib/String.h>

class Module;
typedef Module* (*makeModule)();

typedef AVLTree<clString, makeModule> ModuleSubCategory;
typedef AVLTree<clString, ModuleSubCategory*> ModuleCategory;
typedef AVLTree<clString, ModuleCategory*> ModuleDB;
typedef AVLTreeIter<clString, makeModule> ModuleSubCategoryIter;
typedef AVLTreeIter<clString, ModuleSubCategory*> ModuleCategoryIter;
typedef AVLTreeIter<clString, ModuleCategory*> ModuleDBIter;

class ModuleList {
    ModuleList();
    ~ModuleList();
    static void insert(ModuleSubCategory* subcat, const clString& name,
		       makeModule maker);
    static ModuleCategory* make_category(const clString& cat_name);
    static ModuleSubCategory* make_subcategory(ModuleCategory* cat,
					       const clString& subcat_name);
public:
    static void insert(const clString& category, const clString& subcategory,
		       const clString& name, makeModule maker);
    static void insert(const clString& category,
		       const clString& name, makeModule maker);
    static makeModule lookup(const clString& name);
    static ModuleDB* get_db();
    static ModuleSubCategory* get_all();
};

class RegisterModule {
public:
    RegisterModule(const clString& category, const clString& subcategory,
		   const clString& name, makeModule maker);
    RegisterModule(const clString& category,
		   const clString& name, makeModule maker);
    ~RegisterModule();
};

#endif /* SCI_project_ModuleList_h */
