
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
typedef Module* (*makeModule)(const clString&);

typedef AVLTree<clString, makeModule> ModuleCategory;
typedef AVLTreeIter<clString, makeModule> ModuleCategoryIter;
#ifdef __GNUG__
typedef AVLTree<clString, void*> ModuleDB;
typedef AVLTreeIter<clString, void*> ModuleDBIter;
#else
typedef AVLTree<clString, ModuleCategory*> ModuleDB;
typedef AVLTreeIter<clString, ModuleCategory*> ModuleDBIter;
#endif

class ModuleList {
    static void insert(ModuleCategory* cat, const clString& name,
		       makeModule maker);
    static ModuleCategory* make_category(const clString& cat_name);
protected:
    ModuleList();
    ~ModuleList();
public:
    static void insert(const clString& category,
		       const clString& name, makeModule maker);
    static makeModule lookup(const clString& name);
    static ModuleDB* get_db();
    static ModuleCategory* get_all();
};

class RegisterModule {
public:
    RegisterModule(const clString& category,
		   const clString& name, makeModule maker);
    ~RegisterModule();
};

#endif /* SCI_project_ModuleList_h */
