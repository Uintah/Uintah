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
