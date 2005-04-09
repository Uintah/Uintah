
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

#include <ModuleList.h>
#include <iostream.h>

static ModuleDB* top=0;
static ModuleSubCategory* all_modules=0;

ModuleDB* ModuleList::get_db()
{
    if(!top)
	top=new ModuleDB;
    return top;
}

ModuleSubCategory* ModuleList::get_all()
{
    if(!all_modules)
	all_modules=new ModuleSubCategory;
    return all_modules;
}

void ModuleList::insert(const clString& cat_name, const clString& subcat_name,
			const clString& name, makeModule maker)
{
    ModuleCategory* cat=make_category(cat_name);
    ModuleSubCategory* subcat=make_subcategory(cat, subcat_name);
    insert(subcat, name, maker);
}

void ModuleList::insert(const clString& cat_name,
			const clString& name, makeModule maker)
{
    ModuleCategory* cat=make_category(cat_name);
    ModuleSubCategory* subcat=make_subcategory(cat, "");
    insert(subcat, name, maker);
}

void ModuleList::insert(ModuleSubCategory* subcat,
			const clString& name, makeModule maker)
{
    makeModule old_maker;
    if(subcat->lookup(name, old_maker)){
	cerr << "WARNING: two different modules with the same name: " << name << endl;
    } else {
	subcat->insert(name, maker);
	// Insert it in the master list as well...
	if(!all_modules)
	    all_modules=new ModuleSubCategory;
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
	top=new ModuleDB;
    ModuleCategory* cat;
    if(!top->lookup(cat_name, cat)){
	cat=new ModuleCategory;
	top->insert(cat_name, cat);
    }
    return cat;
}

ModuleSubCategory* ModuleList::make_subcategory(ModuleCategory* cat,
						const clString& subcat_name)
{
    ModuleSubCategory* subcat;
    if(!cat->lookup(subcat_name, subcat)){
	subcat=new ModuleSubCategory;
	cat->insert(subcat_name, subcat);
    }
    return subcat;
}

makeModule ModuleList::lookup(const clString& name)
{
    makeModule ret;
    if(!all_modules)
	ret=0;
    else if(!all_modules->lookup(name, ret)){
	ret=0;
    }
    return ret;
}

RegisterModule::RegisterModule(const clString& category,
			       const clString& subcategory,
			       const clString& name,
			       makeModule maker)
{
    ModuleList::insert(category, subcategory, name, maker);
}

RegisterModule::RegisterModule(const clString& category,
			       const clString& name,
			       makeModule maker)
{
    ModuleList::insert(category, name, maker);
}

RegisterModule::~RegisterModule()
{
}
