
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

class clString;
class Module;
typedef Module* (*makeModule)();

class ModuleList {
    ModuleList();
    ~ModuleList();
public:
    static void insert(const clString& name, makeModule maker);
    static makeModule lookup(const clString& name);
    static void initialize_list();
};

#endif /* SCI_project_ModuleList_h */
