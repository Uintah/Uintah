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
      case Symbol::EnumType:
      case Symbol::DistArrayType:
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

void Symbol::setEnumerator(Enumerator* e)
{
  enumerator=e;
}

Enumerator* Symbol::getEnumerator() const
{
  return enumerator;
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
    {
      string parent_name = parent->cppfullname();
      if( parent_name == "" )
	return name;
      else
	return parent_name + "::" + name;
    }
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

