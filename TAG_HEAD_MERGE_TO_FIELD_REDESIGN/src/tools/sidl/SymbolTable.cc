
#include "SymbolTable.h"
#include "Spec.h"
#include <iostream>

using std::string;
using std::map;
using std::cerr;

SymbolTable::SymbolTable(SymbolTable* parent, const string& name)
    : parent(parent), name(name)
{
}

SymbolTable::~SymbolTable()
{
}

Symbol* SymbolTable::lookup(const string& name, bool recurse) const
{
    map<string, Symbol*>::const_iterator iter=symbols.find(name);
    if(iter == symbols.end()){
	if(recurse && parent)
	    return parent->lookup(name, true);
	else 
	    return 0;
    } else {
	return iter->second;
    }
}

void SymbolTable::insert(Symbol* sym)
{
    sym->setSymbolTable(this);
    symbols[sym->getName()]=sym;
}

void Symbol::setType(Symbol::Type typ)
{
    type=typ;
}

Symbol::Symbol(const string& name)
    : name(name), symtab(0)
{
    definition=0;
    method=0;
    emitted_forward=false;
}

const string& Symbol::getName() const
{
    return name;
}

Symbol* SymbolTable::lookup(ScopedName* name) const
{
    const SymbolTable* p;
    if(name->getLeadingDot()){
	p=this;
	while(p->parent)
	    p=p->parent;
    } else {
	p=this;
    }
    while(p){
	const SymbolTable* pp=p;
	int i=0;
	for(;;){
	    Symbol* s=pp->lookup(name->name(i), false);
	    if(!s)
		break;
	    switch(s->getType()){
	    case Symbol::ClassType:
	    case Symbol::PackageType:
	    case Symbol::InterfaceType:
		{
		    Definition* d=s->getDefinition();
		    pp=d->getSymbolTable();
		}
		break;
	    default:
		return 0;
	    }
	    i++;
	    if(i == name->nnames())
		return s;
	}
	p=p->parent;
    }
    return 0;
}

void Symbol::setDefinition(Definition* def)
{
    definition=def;
}

Symbol::Type Symbol::getType() const
{
    return type;
}

Definition* Symbol::getDefinition() const
{
    return definition;
}

void Symbol::setMethod(Method* m)
{
    method=m;
}

Method* Symbol::getMethod() const
{
    return method;
}

string SymbolTable::fullname() const
{
    if(parent)
	return parent->fullname()+"."+name;
    else
	return "";
}

string SymbolTable::cppfullname() const
{
    if(parent)
	return parent->cppfullname()+"::"+name;
    else
	return "";
}

string Symbol::fullname() const
{
    return symtab->fullname()+"."+name;
}

string Symbol::cppfullname(SymbolTable* forstab) const
{
    if(forstab == symtab)
	return name;
    else
	return symtab->cppfullname()+"::"+name;
}

void Symbol::setSymbolTable(SymbolTable* stab)
{
    if(symtab){
	cerr << "Error: SymbolTable set twice for symbol!\n";
	exit(1);
    }
    symtab=stab;
}

SymbolTable* SymbolTable::getParent() const
{
    return parent;
}

std::string SymbolTable::getName() const
{
    return name;
}

//
// $Log$
// Revision 1.3  1999/09/17 05:07:26  sparker
// Added nexus code generation capability
//
// Revision 1.2  1999/08/30 17:39:54  sparker
// Updates to configure script:
//  rebuild configure if configure.in changes (Bug #35)
//  Fixed rule for rebuilding Makefile from Makefile.in (Bug #36)
//  Rerun configure if configure changes (Bug #37)
//  Don't build Makefiles for modules that aren't --enabled (Bug #49)
//  Updated Makfiles to build sidl and Component if --enable-parallel
// Updates to sidl code to compile on linux
// Imported PIDL code
// Created top-level Component directory
// Added ProcessManager class - a simpler interface to fork/exec (not finished)
//
//
