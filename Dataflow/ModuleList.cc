
/*
 *  ModuleList.cc: Implementation of Module database
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/ModuleList.h>
#include <Classlib/HashTable.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <fstream.h>
#include <string.h>
#include <dlfcn.h>

static ModuleDB* top=0;
static ModuleCategory* all_modules=0;
static HashTable<clString, clString>* homes;
static HashTable<clString, void*>* dlhandles;

ModuleDB* ModuleList::get_db()
{
    if(!top){
	parse_db();
    }
    return top;
}

ModuleCategory* ModuleList::get_all()
{
    if(!all_modules)
	all_modules=scinew ModuleCategory;
    return all_modules;
}

void ModuleList::insert(const clString& cat_name,
			const clString& name, makeModule maker)
{
    ModuleCategory* cat=make_category(cat_name);
    if(!homes)
	homes=new HashTable<clString, clString>;
    clString dummy;
    if(!homes->lookup(name, dummy))
	homes->insert(name, cat_name);
    makeModule old_maker;
    if(cat->lookup(name, old_maker)){
	cerr << "WARNING: two different modules with the same name: " << name << endl;
    } else {
	cat->insert(name, maker);
	// Insert it in the master list as well...
	if(!all_modules)
	    all_modules=scinew ModuleCategory;
	if(all_modules->lookup(name, old_maker)){
	    // Already there, just make sure that the maker's are the same...
	    if(old_maker != maker){
		cerr << "WARNING: two different modules with the same name!!!" << name << endl;
	    }
	} else {
	    all_modules->insert(name, maker);
	}
    }

}

ModuleCategory* ModuleList::make_category(const clString& cat_name)
{
    if(!top)
	top=scinew ModuleDB;
    ModuleCategory* cat;
    if(!top->lookup(cat_name, cat)){
	cat=scinew ModuleCategory;
	top->insert(cat_name, cat);
    }
    return cat;
}

makeModule ModuleList::lookup(const clString& name)
{
    makeModule ret;
    if(!all_modules)
	ret=0;
    else if(!all_modules->lookup(name, ret)){
	ret=0;
    } else {
	if(!ret){
	    // This is a valid module name, but it is zero, so
	    // look it up using dlsym
	    clString dirname;
	    if(!homes || !homes->lookup(name, dirname)){
		cerr << "Cannot find home of module\n";
		return 0; // Can't find it...
	    }
	    if(!dlhandles)
		dlhandles=new HashTable<clString, void*>;
	    void* handle;
	    if(!dlhandles->lookup(dirname, handle)){
		char* home=getenv("SCI_WORK");
		if(!home)home=".";
		clString path(clString("lib")+dirname+".so");
		handle=dlopen(path(), RTLD_LAZY);
		if(!handle){
		    cerr << "Cannot open shared library: " << path << endl;
		    cerr << dlerror() << endl;
		    return 0;
		}
		dlhandles->insert(dirname, handle);
	    }
	    clString makername("make_"+name);
	    void* sym=dlsym(handle, makername());
	    if(!sym){
		cerr << "Cannot locate symbol: " << makername << " in " << dirname << endl;
		cerr << dlerror() << endl;
	    }
	    ret=(makeModule)sym;
	}
    }
    return ret;
}

void ModuleList::parse_db()
{
    clString current_category("Unknown");
    char* home=getenv("SCI_WORK");
    if(!home)home=".";
    clString name(clString(home)+"/MODULES");
    ifstream in(name());
    if(!in){
	cerr << "Cannot open the list of Modules: " << name << endl;
	exit(-1);
    }
    while(in){
	char buf[1000];
	in.getline(buf, 1000);
	if(in){
	    if(strncmp(buf, "Category:", 9) == 0){
		current_category=clString(buf+10);
	    } else if(buf[0] != '#' && buf[0] != '\n' && buf[0] != 0){
		clString modname(buf);
		insert(current_category, modname, 0);
	    }
	}
    }
}

#ifdef __GNUG__
// Template instantiations
#include <Classlib/AVLTree.cc>
template class TreeLink<clString, makeModule>;
template class AVLTree<clString, makeModule>;
template class AVLTreeIter<clString, makeModule>;
template class TreeLink<clString, ModuleCategory*>;
template class AVLTree<clString, ModuleCategory*>;
template class AVLTreeIter<clString, ModuleCategory*>;

#include <Classlib/HashTable.cc>
template class HashTable<clString, clString>;
template class HashKey<clString, clString>;
template class HashTable<clString, void*>;
template class HashKey<clString, void*>;

#endif
