/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



#include <Core/CCA/tools/sidl/Spec.h>
#include <Core/CCA/tools/sidl/SymbolTable.h>
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
    map<string, Symbol*>::iterator iter = symbols.begin();
    while (iter != symbols.end()) {
        delete iter->second;
        symbols.erase(iter);
        iter = symbols.begin();
    }
}

Symbol* SymbolTable::lookup(const string& name, bool recurse) const
{
  map<string, Symbol*>::const_iterator iter=symbols.find(name);
  if(iter == symbols.end()) {
    if(recurse && parent) {
      return parent->lookup(name, true);
    } else {
      return 0;
    }
  } else {
    return iter->second;
  }
}

void SymbolTable::insert(Symbol* sym)
{
  sym->setSymbolTable(this);
  symbols[sym->getName()] = sym;
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

Symbol::~Symbol()
{
}

const string& Symbol::getName() const
{
  return name;
}

Symbol* SymbolTable::lookup(ScopedName* name) const
{
    const SymbolTable* p;
    if (name->getLeadingDot()) {
        p=this;
        while(p->parent) {
            p=p->parent;
        }
    } else {
        p=this;
    }
    while(p) {
        const SymbolTable* pp=p;
        int i=0;
        for(;;) {
            Symbol* s=pp->lookup(name->name(i), false);
            if (!s) {
                break;
            }
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
            if (i == name->nnames()) {
                return s;
            }
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
  if (parent) {
    return parent->fullname()+"."+name;
  } else {
    return "";
  }
}

string SymbolTable::cppfullname() const
{
    if (parent) {
        string parent_name = parent->cppfullname();
        if ( parent_name == "" ) {
            return name;
        } else {
            return parent_name + "::" + name;
        }
    } else {
        return "";
    }
}

string Symbol::fullname() const
{
  return symtab->fullname()+"."+name;
}

string Symbol::cppfullname(SymbolTable* forstab) const
{
  if (forstab == symtab) {
    return name;
  } else {
    return symtab->cppfullname()+"::"+name;
  }
}

void Symbol::setSymbolTable(SymbolTable* stab)
{
  if (symtab) {
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

