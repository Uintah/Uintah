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


#ifndef SYMBOLTABLE_H
#define SYMBOLTABLE_H

#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <sgi_stl_warnings_on.h>

class Definition;
class Method;
class ScopedName;
class SymbolTable;
class EmitState;
class Enumerator;

class Symbol {
public:
  Symbol(const std::string& name);
  ~Symbol();
  
  enum Type {
    ClassType,
    PackageType,
    InterfaceType,
    MethodType,
    EnumType,
    EnumeratorType,
    DistArrayType
  };
  void setType(Type type);
  Type getType() const;
  const std::string& getName() const;
  Definition* getDefinition() const;
  void setDefinition(Definition*);
  Method* getMethod() const;
  void setMethod(Method*);
  Enumerator* getEnumerator() const;
  void setEnumerator(Enumerator*);
  std::string fullname() const;
  std::string cppfullname(SymbolTable* forstab=0) const;
  bool emitted_forward;
  void emit_forward(EmitState& e);
protected:
  friend class SymbolTable;
  void setSymbolTable(SymbolTable*);
  void emit(EmitState& e);
private:
  Type type;
  std::string name;
  Definition* definition;
  Method* method;
  SymbolTable* symtab;
  Enumerator* enumerator;
};

class SymbolTable {
  SymbolTable* parent;
  std::map<std::string, Symbol*> symbols;
  std::string name;
public:
  SymbolTable(SymbolTable* parent, const std::string& name);
  ~SymbolTable();

  Symbol* lookup(const std::string& name, bool recurse) const;
  Symbol* lookup(ScopedName* name) const;
  Symbol* lookupScoped(const std::string& name) const;
  void insert(Symbol*);

  SymbolTable* getParent() const;
  std::string getName() const;

  std::string fullname() const;
  std::string cppfullname() const;
  void emit(EmitState& e) const;

};

#endif
