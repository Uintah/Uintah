
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
typedef Module* (*makeModule)(const clString&, int remote=0);

typedef AVLTree<clString, makeModule> ModuleCategory;
typedef AVLTreeIter<clString, makeModule> ModuleCategoryIter;
typedef AVLTree<clString, ModuleCategory*> ModuleDB;
typedef AVLTreeIter<clString, ModuleCategory*> ModuleDBIter;

class ModuleList {
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
    static void parse_db();
};

#endif /* SCI_project_ModuleList_h */
