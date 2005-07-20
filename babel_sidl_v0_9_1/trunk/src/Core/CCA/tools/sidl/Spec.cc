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
#include <map>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>

using std::cerr;
using std::find;
using std::map;
using std::string;
using std::vector;
using std::pair;
extern bool foremit;

Definition::Definition(const string& curfile, int lineno, const string& name)
  : curfile(curfile), lineno(lineno), name(name)
{
  symbols=0;
  emitted_declaration=false;
  do_emit=::foremit;
  isImport=false;
  isBuiltin = false;
}

void Definition::setName(const std::string& n)
{
  name=n;
}

Definition::~Definition()
{
}

string Definition::fullname() const
{
  if (!symbols) {
    cerr << "ERROR: Definition SymbolTable not set!\n";
    exit(1);
  }
  return symbols->fullname();
}

string CI::cppfullname(SymbolTable* forpackage) const
{
  if (!symbols){
    cerr << "ERROR: Definition SymbolTable not set!\n";
    exit(1);
  }
  if (forpackage == symbols->getParent()) {
    return name;
  } else {
    return symbols->cppfullname();
  }
}

string CI::cppclassname() const
{
  return name;
}

SymbolTable* Definition::getSymbolTable() const
{
  return symbols;
}

Argument::Argument(bool copy, Argument::Mode mode, Type* type)
  : copy(copy), mode(mode), type(type)
{
}

Argument::Argument(bool copy, Argument::Mode mode, Type* type, const string& id)
  : copy(copy), mode(mode), type(type), id(id)
{
}

Argument::~Argument()
{
}

string Argument::fullsignature() const
{
  string s;
  switch(mode){
  case In:
    s="in ";
    break;
  case Out:
    s="out ";
    break;
  case InOut:
    s="inout ";
    break;
  }
  s+=type->fullname();
  if(id != "")
    s+=" "+id;
  return s;
}

Argument::Mode Argument::getMode() const
{
  return mode;
}

Type* Argument::getType() const
{
  return type;
}

ArgumentList::ArgumentList()
{
}

ArgumentList::~ArgumentList()
{
  for(vector<Argument*>::iterator iter=list.begin(); iter != list.end();iter++){
    delete *iter;
  }
}

void ArgumentList::prepend(Argument* arg)
{
  list.insert(list.begin(), arg);
}

void ArgumentList::add(Argument* arg)
{
  list.push_back(arg);
}

std::vector<Argument*>& ArgumentList::getList()
{
  return list;
}

string ArgumentList::fullsignature() const
{
    std::string s;
    if (list.empty()) {
        return s;
    }
    int c = 0;
    for (vector<Argument*>::const_iterator iter = list.begin();
            iter != list.end(); iter++) {
        if(c++ > 0) {
            s+=", ";
        }
        s+=(*iter)->fullsignature();
    }
    return s;
}

CI::CI(const string& curfile, int lineno, const string& name,
       MethodList* methods, DistributionArrayList* arrays)
  : Definition(curfile, lineno, name), mymethods(methods), mydistarrays(arrays)
{
  doRedistribution = false;
  exceptionID = 0;
  vtable_base = -1234;
  parentclass = 0;
}

CI::~CI()
{
  if (mymethods) {
    delete mymethods;
  }
  if (mydistarrays) {
    delete mydistarrays;
  }
}

Class::Class(const string& curfile, int lineno, Modifier modifier,
	     const string& name,  ScopedName* class_extends,
	     ScopedNameList* class_implementsall,
	     ScopedNameList* class_implements,
	     MethodList* methods,
	     DistributionArrayList* arrays)
  : CI(curfile, lineno, name, methods,arrays),
    modifier(modifier),
    class_extends(class_extends), class_implementsall(class_implementsall),
    class_implements(class_implements)
{
  std::cerr << "Class::Class: curfile=" << curfile << " name=" << name << std::endl;
if (class_extends) {
    // hack to support CCAException extends SSIDL.BaseException
    std::cerr << " class extends=" << class_extends->getName() << std::endl;
    if (name == "CCAException" && class_extends->getName() == "SSIDL.BaseException") {
        std::cerr << "\tHave CCAException extends SSIDL.BaseException" << std::endl;
        delete class_extends;
        class_extends = 0;
        class_extends = new ScopedName("SSIDL", "SIDLException");
    }
}
if (class_implements) {
std::vector<ScopedName*> list = class_implements->getList();
for (vector<ScopedName*>::const_iterator iter=list.begin(); iter != list.end(); iter++) {
  std::cerr << "  class implements=" << (*iter)->getName() << std::endl;
}
}
if (class_implementsall) {
std::vector<ScopedName*> list = class_implementsall->getList();
for (vector<ScopedName*>::const_iterator iter=list.begin(); iter != list.end(); iter++) {
  std::cerr << "  class implementsall=" << (*iter)->getName() << std::endl;
}
}
}

Class::~Class()
{
  if (class_extends) {
    delete class_extends;
  }
  if (class_implements) {
    delete class_implements;
  }
  if (class_implementsall) {
    delete class_implementsall;
  }
}

void CI::gatherMethods(vector<Method*>& allmethods) const
{
  if (parentclass) {
    parentclass->gatherMethods(allmethods);
  }
  for (vector<BaseInterface*>::const_iterator iter = parent_ifaces.begin();
       iter != parent_ifaces.end(); iter++) {
std::cerr << "CI::gatherMethods: parent_ifaces " << (*iter)->cppclassname() << std::endl;
    (*iter)->gatherMethods(allmethods);
  }
  mymethods->gatherMethods(allmethods);
}

void CI::gatherVtable(vector<Method*>& uniquemethods, bool onlyNewMethods) const
{
  if (parentclass) {
    parentclass->gatherVtable(uniquemethods, false);
  }
  for (vector<BaseInterface*>::const_iterator iter = parent_ifaces.begin();
       iter != parent_ifaces.end(); iter++) {
std::cerr << "CI::gatherVtable: parent_ifaces " << (*iter)->cppclassname() << std::endl;
    (*iter)->gatherVtable(uniquemethods, false);
  }
  int s = uniquemethods.size();
  mymethods->gatherVtable(uniquemethods);
  if (onlyNewMethods) {
    uniquemethods.erase(uniquemethods.begin(), uniquemethods.begin()+s);
  }
}

vector<Method*>& CI::myMethods()
{
  return mymethods->getList();
}

void CI::gatherParents(std::vector<CI*>& folks) const
{
  if (parentclass) {
    parentclass->gatherParents(folks);
  }
  for (vector<BaseInterface*>::const_iterator iter = parent_ifaces.begin();
       iter != parent_ifaces.end(); iter++) {
std::cerr << "CI::gatherParents: parent_ifaces " << (*iter)->cppclassname() << std::endl;
    (*iter)->gatherParents(folks);
  }
  if (find(folks.begin(), folks.end(), this) == folks.end()) {
std::cerr << "CI::gatherParents: folks " << this->cppclassname() << std::endl;
    folks.push_back(const_cast<CI*>(this));
  }
}

void CI::gatherParentInterfaces(std::vector<BaseInterface*>& folks) const
{
  if (parentclass) {
    parentclass->gatherParentInterfaces(folks);
  }
  for (vector<BaseInterface*>::const_iterator iter = parent_ifaces.begin();
       iter != parent_ifaces.end(); iter++){
std::cerr << "CI::gatherParentInterfaces: parent_ifaces " << (*iter)->cppclassname() << std::endl;
    (*iter)->gatherParentInterfaces(folks);
    if (find(folks.begin(), folks.end(), *iter) == folks.end()) {
std::cerr << "CI::gatherParentInterfaces: folks " << (*iter)->cppclassname() << std::endl;
      folks.push_back(const_cast<BaseInterface*>(*iter));
    }
  }
}

MethodList* CI::getMethods() const
{
  return mymethods;
}

Class* Class::getParentClass() const
{
  return parentclass;
}

BaseInterface* Class::findParentInterface(const string& str)
{
std::cerr << "Class::findParentInterface: " << str << std::endl;
if (class_implements) {
std::cerr << "Class::findParentInterface: class_implements fullsignature=" << class_implements->fullsignature()  << std::endl;
}
if (class_implementsall) {
std::cerr << "Class::findParentInterface: class_implementsall fullsignature=" << class_implementsall->fullsignature()  << std::endl;
}
    return 0;
}

Class* Class::findParent(const string& str)
{
  Class* p=this;
  while(p){
    if(p->fullname() == str)
      return p;
    p=p->parentclass;
if (p) {
std::cerr << "Class::findParent: parent=" << p->getName() << std::endl;
} else {
std::cerr << "Class::findParent: null parent" << std::endl;
}
  }
  return 0;
}

DefinitionList::DefinitionList()
{
  isBuiltin = false;
}

DefinitionList::~DefinitionList()
{
  for (vector<Definition*>::iterator iter = list.begin(); iter != list.end(); iter++) {
    delete *iter;
  }
}

void DefinitionList::add(Definition* def)
{
  list.push_back(def);
}

void DefinitionList::processImports()
{
  for (vector<Definition*>::iterator iter = list.begin(); iter != list.end(); iter++) {
    (*iter)->isImport = true;
    Package* pkg;
    if ((pkg = dynamic_cast<Package*>(*iter))) {
std::cerr << "DefinitionList::processImports: " << pkg->getName() << std::endl;
        pkg->definition->processImports();
    }
  }
}

// setting parent interfaces or classes for interfaces???
BaseInterface::BaseInterface(const string& curfile, int lineno, const string& id,
		     ScopedNameList* interface_extends, MethodList* methods,
		     DistributionArrayList* arrays)
  : CI(curfile, lineno, id, methods,arrays),
    interface_extends(interface_extends)
{
std::cerr << "BaseInterface::BaseInterface: " << curfile << " " << lineno << " " << id << std::endl;
if (interface_extends) {
std::vector<ScopedName*> list = interface_extends->getList();
for (vector<ScopedName*>::const_iterator iter=list.begin(); iter != list.end(); iter++) {
  std::cerr << "  interface extends=" << (*iter)->getName() << std::endl;
}
}
}


BaseInterface::~BaseInterface()
{
    if (interface_extends) {
        delete interface_extends;
    }
}

Method::Method(const string& curfile, int lineno, bool copy_return,
               Type* return_type, const string& name, const std::string &babel_ext,
               ArgumentList* args, Modifier2 modifier2, ScopedNameList* throws_clause)
  : curfile(curfile), lineno(lineno), copy_return(copy_return),
    return_type(return_type), name(name), babel_ext(babel_ext), args(args),
    modifier2(modifier2), throws_clause(throws_clause), modifier(Unknown)
{
  doRedistribution = false;
  myclass = 0;
  myinterface = 0;
  checked = false;
  handlerNum =- 1;
  std::cerr << "Method::Method: " << curfile << " " << lineno << " " << name << " " << babel_ext << std::endl;
}

Method::Method(const string& curfile, int lineno, bool copy_return,
	       Type* return_type, const string& name, ArgumentList* args,
	       Modifier2 modifier2, ScopedNameList* throws_clause)
  : curfile(curfile), lineno(lineno), copy_return(copy_return),
    return_type(return_type), name(name), args(args),
    modifier2(modifier2), throws_clause(throws_clause), modifier(Unknown)
{
  doRedistribution = false;
  myclass = 0;
  myinterface = 0;
  checked = false;
  handlerNum =- 1;
  std::cerr << "Method::Method: " << curfile << " " << lineno << " " << name << std::endl;
}

Method::~Method()
{
  delete args;
  delete throws_clause;
}

void Method::setModifier(Method::Modifier newmodifier)
{
  modifier=newmodifier;
}

Method::Modifier Method::getModifier() const
{
  return modifier;
}

void Method::setClass(Class *c)
{
std::cerr << "Method::setClass: " << c->getName() << std::endl;
  myclass=c;
}

void Method::setInterface(BaseInterface *c)
{
std::cerr << "Method::setInterface: " << c->getName() << std::endl;
  myinterface=c;
}

bool Method::getChecked() const
{
  return checked;
}

const string& Method::getCurfile() const
{
  return curfile;
}

int Method::getLineno() const
{
  return lineno;
}

string Method::fullname() const
{
  string n;
  if (myclass) {
    n = myclass->fullname();
  } else if(myinterface) {
    n = myinterface->fullname();
  } else {
    cerr << "ERROR: Method does not have an associated class or interface\n";
    exit(1);
  }
  return n+"."+name;
}

string Method::fullsignature() const
{
  string s;
  switch(modifier){
  case Abstract:
    s="abstract ";
    break;
  case Final:
    s="final ";
    break;
  case Static:
    s="static ";
    break;
  case Unknown:
  case NoModifier:
    s="";
    break;
  }
  s+=return_type->fullname()+" "+fullname()+"("+args->fullsignature()+")";
  if(throws_clause)
    s+="throws "+throws_clause->fullsignature();
  return s;
}

MethodList::MethodList()
{
}

MethodList::~MethodList()
{
    for (vector<Method*>::iterator iter=list.begin(); iter != list.end(); iter++) {
        delete *iter;
    }
}

void MethodList::add(Method* method)
{
//std::cerr <<  "MethodList::add " << method->getName() << std::endl;
  list.push_back(method);
}

void MethodList::setClass(Class* c)
{
//std::cerr <<  "MethodList::setClass " << c->getName() << std::endl;
    for (vector<Method*>::iterator iter = list.begin(); iter != list.end(); iter++) {
        (*iter)->setClass(c);
    }
}

void MethodList::setInterface(BaseInterface* c)
{
std::cerr <<  "MethodList::setInterface " << c->getName() << std::endl;
  for(vector<Method*>::iterator iter=list.begin();iter != list.end();iter++) {
    (*iter)->setInterface(c);
std::cerr << "MethodList::setInterface: BaseInterface " << (*iter)->getName() << std::endl;
  }
}

vector<Method*>& MethodList::getList()
{
  return list;
}

void MethodList::gatherMethods(vector<Method*>& allmethods) const
{
  for (vector<Method*>::const_iterator iter=list.begin();iter != list.end();iter++) {
    allmethods.push_back(*iter);
//std::cerr << "MethodList::gatherMethods: all methods " << (*iter)->fullname() << std::endl;
  }
}

void MethodList::gatherVtable(vector<Method*>& uniquemethods) const
{
  // Yuck - O(N^2)
  for(vector<Method*>::const_iterator iter1=list.begin();iter1 != list.end();iter1++){
    vector<Method*>::iterator iter2=uniquemethods.begin();
    for(;iter2 != uniquemethods.end(); iter2++) {
      if((*iter1)->matches(*iter2, Method::TypesOnly)) {
          break;
      }
    }
    if (iter2 == uniquemethods.end()) {
      uniquemethods.push_back(*iter1);
    }
  }
}

Package::Package(const string& fileno, int lineno, const std::string& name,
		 DefinitionList* definition)
  : Definition(fileno, lineno, name), definition(definition)
{
}

Package::~Package()
{
    if (definition) {
        delete definition;
    }
}

ScopedName::ScopedName()
  : leading_dot(false)
{
//std::cerr << "ScopedName::ScopedName" << std::endl;
  sym=0;
}

ScopedName::ScopedName(const string& s1, const string& s2)
{
//std::cerr << "ScopedName::ScopedName s1=" << s1 << " s2=" << s2 << std::endl;
  leading_dot=true;
  add(s1);
  add(s2);
  sym=0;
}

ScopedName::ScopedName(const string& s1)
{
//std::cerr << "ScopedName::ScopedName s1=" << s1 << std::endl;
  leading_dot=false;
  add(s1);
  sym=0;
}

ScopedName::~ScopedName()
{
}

void ScopedName::prepend(const string& name)
{
  names.insert(names.begin(), name);
}

void ScopedName::add(const string& name)
{
  names.push_back(name);
}

void ScopedName::set_leading_dot(bool dot)
{
  leading_dot=dot;
}

string ScopedName::getName() const
{
  string n;
  if (leading_dot) {
    n+=".";
  }
  bool first=true;
  for(vector<string>::const_iterator iter=names.begin();iter != names.end();iter++) {
    if (!first) {
      n+=".";
    }
    n+=*iter;
    first=false;
  }
  return n;
}

Symbol* ScopedName::getSymbol() const
{
  return sym;
}

bool ScopedName::getLeadingDot() const
{
  return leading_dot;
}

const string& ScopedName::name(int idx) const
{
  return names[idx];
}

int ScopedName::nnames() const
{
  return names.size();
}

string ScopedName::fullname() const
{
  string n;
  if (!sym) {
    cerr << "ERROR: Symbol not bound: " << getName() << " - " << this << "(ScopedName::fullname)\n";
    exit(1);
  } else {
    n = sym->fullname();
  }
  return n;
}

string ScopedName::cppfullname(SymbolTable* forstab) const
{
  string n;
  if (!sym) {
    cerr << "ERROR: Symbol not bound: " << getName() << " - " << this << " (ScopedName::cppfullname)\n";
    exit(1);
  } else {
    n = sym->cppfullname(forstab);
  }
  return n;
}

ScopedNameList::ScopedNameList()
{
}

ScopedNameList::~ScopedNameList()
{
    for (vector<ScopedName*>::iterator iter = list.begin();
        iter != list.end(); iter++) {
        delete *iter;
    }
}

void ScopedNameList::prepend(ScopedName* name)
{
std::cerr << "ScopedNameList::prepend: " << name->getName() << std::endl;
    list.insert(list.begin(), name);
}

void ScopedNameList::add(ScopedName* name)
{
//std::cerr << "ScopedNameList::add: " << name->getName() << std::endl;
  list.push_back(name);
}

vector<ScopedName*> const& ScopedNameList::getList() const
{
  return list;
}

std::string ScopedNameList::fullsignature() const
{
    string s;
    if (list.empty()) {
        return s;
    }
    int c = 0;
    for (vector<ScopedName*>::const_iterator iter = list.begin();
            iter != list.end(); iter++) {
        if (c++ > 0) {
            s += ", ";
        }
        s += (*iter)->fullname();
    }
    return s;
}

Specification::Specification(VersionList* versions, ScopedNameList* imports,
			     DefinitionList* packages)
  : versions(versions), imports(imports), packages(packages)
{
  isImport = false;
  isBuiltin = false;
  toplevel = false;
}

Specification::~Specification()
{
  if (versions) {
    delete versions;
  }
  if (imports) {
    delete imports;
  }
  if (packages) {
    delete packages;
  }
}

void Specification::setTopLevel()
{
  toplevel = true;
}

Type* Type::voidtype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("void", "error");
  }
  return t;
}

Type* Type::booltype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("bool", "byte");
  }
  return t;
}

Type* Type::chartype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("char", "char");
  }
  return t;
}

Type* Type::dcomplextype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("::std::complex<double> ", "special");
  }
  return t;
}

Type* Type::doubletype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("double", "double");
  }
  return t;
}

Type* Type::fcomplextype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("::std::complex<float> ", "special");
  }
  return t;
}

Type* Type::floattype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("float", "float");
  }
  return t;
}

Type* Type::inttype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("int", "int");
  }
  return t;
}

Type* Type::longtype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("long", "long");
  }
  return t;
}

Type* Type::opaquetype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("void*", "opaque");
  }
  return t;
}

Type* Type::stringtype()
{
  static Type* t;
  if (!t) {
    t = new BuiltinType("string", "special");
  }
  return t;
}

static map<pair<Type*, int>, Type*> arrays;

Type* Type::arraytype(Type* t, int dim)
{
  if (dim == 0) {
    dim = 1;
  }
  map<pair<Type*, int>, Type*>::iterator iter=arrays.find(pair<Type*, int>(t, dim));
  if (iter == arrays.end()) {
    return (arrays[pair<Type*, int>(t, dim)]=new ArrayType(t, dim));
  } else {
    return iter->second;
  }
}

Type::Type()
{
}

Type::~Type()
{
}

BuiltinType::BuiltinType(const string& cname, const string& nexusname)
  : cname(cname), nexusname(nexusname)
{
}

BuiltinType::~BuiltinType()
{
}

std::string BuiltinType::fullname() const
{
  return cname;
}

std::string BuiltinType::cppfullname(SymbolTable*) const
{
  if(cname == "string") {
    return "::std::string";
  } else {
    return cname;
  }
}

bool BuiltinType::isvoid() const
{
  if(cname == "void") {
    return true;
  } else {
    return false;
  }
}

NamedType::NamedType(const string& curfile, int lineno, ScopedName* name)
  : curfile(curfile), lineno(lineno), name(name)
{
}

NamedType::~NamedType()
{
}

std::string NamedType::fullname() const
{
  return name->fullname();
}

bool NamedType::isvoid() const
{
  return false;
}

std::string NamedType::cppfullname(SymbolTable* localScope) const
{
  Symbol::Type symtype = name->getSymbol()->getType();
  if (symtype == Symbol::EnumType) {
    return name->cppfullname(localScope);
  } else if(name->getSymbol()->getDefinition()->isEmitted()) {
    return name->cppfullname(localScope)+"::pointer";
  } else {
    return "CCALib::SmartPointer<"+name->cppfullname(localScope)+" >";
  }
}

ArrayType::ArrayType(Type* t, int dim)
  : dim(dim), subtype(t)
{
}

ArrayType::~ArrayType()
{
}

std::string ArrayType::fullname() const
{
  std::ostringstream o;
  o << "array" << dim << "< " << subtype->fullname() << ", " << dim << ">";
  return o.str();
}

std::string ArrayType::cppfullname(SymbolTable* localScope) const
{
  std::ostringstream o;
  o << "::SSIDL::array" << dim << "< " << subtype->cppfullname(localScope);
  if (dynamic_cast<ArrayType*>(subtype) || dynamic_cast<NamedType*>(subtype)) {
    o << " "; // Keep > > from being >>
  }
  o << ">";
  return o.str();
}

bool ArrayType::isvoid() const
{
  return false;
}

Enum::Enum(const std::string& curfile, int lineno,
	   const std::string& name)
  : Definition(curfile, lineno, name)
{
std::cerr << "Enum::Enum: " << name << std::endl;
}

Enum::~Enum()
{
  for(vector<Enumerator*>::iterator iter = list.begin();
      iter != list.end(); iter++)
    delete *iter;
}

void Enum::add(Enumerator* e)
{
  list.push_back(e);
}

Enumerator::Enumerator(const std::string& curfile, int lineno,
		       const std::string& name)
  : curfile(curfile), lineno(lineno), name(name), have_value(false)
{
std::cerr << "Enumerator::Enumerator: " << name << std::endl;
}

Enumerator::Enumerator(const std::string& curfile, int lineno,
		       const std::string& name, int value)
  : curfile(curfile), lineno(lineno), name(name), have_value(true),
    value(value)
{
std::cerr << "Enumerator::Enumerator: have value, " << name << std::endl;
}

Enumerator::~Enumerator()
{
}

VersionList::VersionList()
{
}

VersionList::~VersionList()
{
  for(vector<Version*>::iterator iter = list.begin();
      iter != list.end(); iter++)
    delete *iter;
}

void VersionList::add(Version* v)
{
  list.push_back(v);
}

Version::Version(const std::string& id, int v)
  : id(id)
{
std::cerr << "Version::Version(string, int): " << id << ", " << v << std::endl;
  std::ostringstream o;
  o << v;
  version=o.str();
}

Version::Version(const std::string& id, const std::string& version)
  : id(id), version(version)
{
std::cerr << "Version::Version(string, string): " << id << ", " << version << std::endl;
}

Version::~Version()
{
}

SpecificationList::SpecificationList()
{
}

SpecificationList::~SpecificationList()
{
  for(vector<Specification*>::iterator iter = specs.begin();
      iter != specs.end(); iter++){
    delete *iter;
  }
}

void SpecificationList::add(Specification* spec)
{
  specs.push_back(spec);
}

void SpecificationList::processImports()
{
std::cerr << "SpecificationList::processImports" << std::endl;
  for(vector<Specification*>::iterator iter = specs.begin();
      iter != specs.end(); iter++){
    if ((*iter)->isImport) {
      (*iter)->packages->processImports();
    }
  }
	        
  // "processImports not finished!\n" -- Might want to remove unnecessary packages here;
}



DistributionArray::DistributionArray(const std::string& curfile, int lineno, const std::string& name, ArrayType* arr_t)
  : Definition(curfile,lineno,name), array_t(arr_t), name(name)
{
}

DistributionArray::~DistributionArray() 
{ 
    if (array_t) {
        delete array_t;
    }
}

ArrayType* DistributionArray::getArrayType() 
{
  return array_t;
}

std::string DistributionArray::getName() 
{
  return name;
}

DistributionArrayList::DistributionArrayList()
{
}

DistributionArrayList::~DistributionArrayList()
{
  for(vector<DistributionArray*>::iterator iter=list.begin();iter != list.end();iter++){
    delete *iter;
  }
}

void DistributionArrayList::add(DistributionArray* d)
{
  list.push_back(d);
}



