
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
#include <Classlib/HashTable.h>
#include <Classlib/String.h>

static HashTable<clString, makeModule>* index;

void ModuleList::insert(const clString& name, makeModule maker)
{
    if(!index)
	index=new HashTable<clString, makeModule>;
    index->insert(name, maker);
}

makeModule ModuleList::lookup(const clString& name)
{
    makeModule ret;
    if(!index->lookup(name, ret)){
	ret=0;
    }
    return ret;
}
