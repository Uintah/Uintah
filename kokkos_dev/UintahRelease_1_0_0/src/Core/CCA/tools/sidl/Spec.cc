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

extern bool foremit;

// whitespace at the end is needed to generate nested templates correctly
const std::string BuiltinType::COMPLEX_FLOAT_NAME("complex<float> ");
const std::string BuiltinType::COMPLEX_DOUBLE_NAME("complex<double> ");

Definition::Definition(const std::string& curfile, int lineno, const std::string& name)
  : curfile(curfile), lineno(lineno), name(name)
{
  symbols = 0;
  emitted_declaration = false;
  do_emit = ::foremit;
  isImport = false;
}

void Definition::setName(const std::string& n)
{
  name = n;
}

Definition::~Definition()
{
}

std::string Definition::fullname() const
{
  if (!symbols) {
    std::cerr << "ERROR: Definition SymbolTable not set! " << __FILE__ << ", " <<
            __LINE__ << std::endl;
    exit(1);
  }
  return symbols->fullname();
}

std::string CI::cppfullname(SymbolTable* forpackage) const
{
  if (!symbols){
    std::cerr << "ERROR: Definition SymbolTable not set! " << __FILE__ << ", " <<
         __LINE__ << std::endl;
    exit(1);
  }
  if (forpackage == symbols->getParent()) {
    return name;
  } else {
    return symbols->cppfullname();
  }
}

std::string CI::cppclassname() const
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

Argument::Argument(bool copy, Argument::Mode mode, Type* type, const std::string& id)
  : copy(copy), mode(mode), type(type), id(id)
{
}

Argument::~Argument()
{
}

std::string Argument::fullsignature() const
{
  std::string s;
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
  for(std::vector<Argument*>::iterator iter=list.begin(); iter != list.end();iter++){
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

std::string ArgumentList::fullsignature() const
{
    std::string s;
    if (list.empty()) {
        return s;
    }
    int c = 0;
    for (std::vector<Argument*>::const_iterator iter = list.begin();
            iter != list.end(); iter++) {
        if (c++ > 0) {
            s += ", ";
        }
        s += (*iter)->fullsignature();
    }
    return s;
}

CI::CI(const std::string& curfile, int lineno, const std::string& name,
       MethodList* methods, DistributionArrayList* arrays)
  : Definition(curfile, lineno, name), mymethods(methods), mydistarrays(arrays)
{
  castException_emitted = false;
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

Class::Class(const std::string& curfile, int lineno, Modifier modifier,
	     const std::string& name,  ScopedName* class_extends,
	     ScopedNameList* class_implementsall,
	     ScopedNameList* class_implements,
	     MethodList* methods,
	     DistributionArrayList* arrays)
  : CI(curfile, lineno, name, methods,arrays),
    modifier(modifier),
    class_extends(class_extends), class_implementsall(class_implementsall),
    class_implements(class_implements)
{
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

void CI::gatherMethods(std::vector<Method*>& allmethods) const
{
  if (parentclass) {
    parentclass->gatherMethods(allmethods);
  }
  for (std::vector<BaseInterface*>::const_iterator iter = parent_ifaces.begin();
       iter != parent_ifaces.end(); iter++) {
    (*iter)->gatherMethods(allmethods);
  }
  mymethods->gatherMethods(allmethods);
}

void CI::gatherVtable(std::vector<Method*>& uniquemethods, bool onlyNewMethods) const
{
  if (parentclass) {
    parentclass->gatherVtable(uniquemethods, false);
  }
  for (std::vector<BaseInterface*>::const_iterator iter = parent_ifaces.begin();
       iter != parent_ifaces.end(); iter++) {
    (*iter)->gatherVtable(uniquemethods, false);
  }
  int s = uniquemethods.size();
  mymethods->gatherVtable(uniquemethods);
  if (onlyNewMethods) {
    uniquemethods.erase(uniquemethods.begin(), uniquemethods.begin()+s);
  }
}

std::vector<Method*>& CI::myMethods()
{
  return mymethods->getList();
}

void CI::gatherParents(std::vector<CI*>& folks) const
{
  if (parentclass) {
    parentclass->gatherParents(folks);
  }
  for (std::vector<BaseInterface*>::const_iterator iter = parent_ifaces.begin();
       iter != parent_ifaces.end(); iter++) {
    (*iter)->gatherParents(folks);
  }
  if (find(folks.begin(), folks.end(), this) == folks.end()) {
    folks.push_back(const_cast<CI*>(this));
  }
}

void CI::gatherParentInterfaces(std::vector<BaseInterface*>& folks) const
{
  if (parentclass) {
    parentclass->gatherParentInterfaces(folks);
  }
  for (std::vector<BaseInterface*>::const_iterator iter = parent_ifaces.begin();
       iter != parent_ifaces.end(); iter++){
    (*iter)->gatherParentInterfaces(folks);
    if (find(folks.begin(), folks.end(), *iter) == folks.end()) {
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

Class* Class::findParent(const std::string& str)
{
  Class* p = this;
  while(p) {
    if (p->fullname() == str) {
      return p;
    }
    p = p->parentclass;
  }
  return 0;
}

DefinitionList::DefinitionList()
{
}

DefinitionList::~DefinitionList()
{
  for (std::vector<Definition*>::iterator iter = list.begin(); iter != list.end(); iter++) {
    delete *iter;
  }
}

void DefinitionList::add(Definition* def)
{
  list.push_back(def);
}

void DefinitionList::processImports()
{
  for (std::vector<Definition*>::iterator iter = list.begin(); iter != list.end(); iter++) {
    (*iter)->isImport = true;
    Package* pkg;
    if ((pkg = dynamic_cast<Package*>(*iter))) {
        pkg->definition->processImports();
    }
  }
}

// setting parent interfaces or classes for interfaces???
BaseInterface::BaseInterface(const std::string& curfile, int lineno, const std::string& id,
		     ScopedNameList* interface_extends, MethodList* methods,
		     DistributionArrayList* arrays)
  : CI(curfile, lineno, id, methods, arrays),
    interface_extends(interface_extends)
{
}


BaseInterface::~BaseInterface()
{
    if (interface_extends) {
        delete interface_extends;
    }
}

BaseInterface* BaseInterface::findParent(const std::string& str)
{
    for (std::vector<BaseInterface*>::iterator iter = parent_ifaces.begin();
            iter != parent_ifaces.end(); iter++) {
        if ((*iter)->fullname() == str) {
            return *iter;
        } else {
            BaseInterface *bi = (*iter)->findParent(str);
            if (bi) {
                return bi;
            }
        }
    }
    return 0;
}

Method::Method(const std::string& curfile, int lineno, bool copy_return,
               Type* return_type, const std::string& name, const std::string &babel_ext,
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
}

Method::Method(const std::string& curfile, int lineno, bool copy_return,
	       Type* return_type, const std::string& name, ArgumentList* args,
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
}


Method::~Method()
{
    if (args) {
        delete args;
    }
    if (throws_clause) {
        delete throws_clause;
    }
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
  myclass = c;
}

void Method::setInterface(BaseInterface *c)
{
  myinterface = c;
}

bool Method::getChecked() const
{
  return checked;
}

const std::string& Method::getCurfile() const
{
  return curfile;
}

int Method::getLineno() const
{
  return lineno;
}

std::string Method::fullname() const
{
  std::string n;
  if (myclass) {
    n = myclass->fullname();
  } else if (myinterface) {
    n = myinterface->fullname();
  } else {
    std::cerr << "ERROR: Method " << name << " does not have an associated class or interface" << std::endl;
    exit(1);
  }
  return n + "." + name;
}

std::string Method::fullsignature() const
{
  std::string s;
  switch(modifier) {
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
  s += return_type->fullname() + " " + fullname() + "(" + args->fullsignature() + ")";
  if (throws_clause)
    s += " throws " + throws_clause->fullsignature();
  return s;
}

MethodList::MethodList()
{
}

MethodList::~MethodList()
{
    for (std::vector<Method*>::iterator iter=list.begin(); iter != list.end(); iter++) {
        delete *iter;
    }
}

void MethodList::add(Method* method)
{
  list.push_back(method);
}

void MethodList::setClass(Class* c)
{
    for (std::vector<Method*>::iterator iter = list.begin(); iter != list.end(); iter++) {
        (*iter)->setClass(c);
    }
}

void MethodList::setInterface(BaseInterface* c)
{
  for(std::vector<Method*>::iterator iter=list.begin();iter != list.end();iter++) {
    (*iter)->setInterface(c);
  }
}

std::vector<Method*>& MethodList::getList()
{
  return list;
}

void MethodList::gatherMethods(std::vector<Method*>& allmethods) const
{
  for (std::vector<Method*>::const_iterator iter=list.begin();iter != list.end();iter++) {
    allmethods.push_back(*iter);
  }
}

void MethodList::gatherVtable(std::vector<Method*>& uniquemethods) const
{
  // Yuck - O(N^2)
  for (std::vector<Method*>::const_iterator iter1 = list.begin(); iter1 != list.end(); iter1++) {
    std::vector<Method*>::iterator iter2 = uniquemethods.begin();
    for (; iter2 != uniquemethods.end(); iter2++) {
      if ((*iter1)->matches(*iter2, Method::TypesOnly)) {
          break;
      }
    }
    if (iter2 == uniquemethods.end()) {
      uniquemethods.push_back(*iter1);
    }
  }
}

Package::Package(const std::string& fileno, int lineno, const std::string& name,
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
  sym = 0;
}

ScopedName::ScopedName(const std::string& s1, const std::string& s2)
{
  leading_dot = true;
  add(s1);
  add(s2);
  sym = 0;
}

ScopedName::ScopedName(const std::string& s1)
{
  leading_dot = false;
  add(s1);
  sym = 0;
}

ScopedName::~ScopedName()
{
}

void ScopedName::prepend(const std::string& name)
{
  names.insert(names.begin(), name);
}

void ScopedName::add(const std::string& name)
{
  names.push_back(name);
}

void ScopedName::set_leading_dot(bool dot)
{
  leading_dot=dot;
}

std::string ScopedName::getName() const
{
  std::string n;
  if (leading_dot) {
    n += ".";
  }
  bool first=true;
  for (std::vector<std::string>::const_iterator iter = names.begin(); iter != names.end(); iter++) {
    if (!first) {
      n += ".";
    }
    n += *iter;
    first = false;
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

const std::string& ScopedName::name(int idx) const
{
  return names[idx];
}

int ScopedName::nnames() const
{
  return names.size();
}

std::string ScopedName::fullname() const
{
  if (!sym) {
    std::cerr << "ERROR: Symbol not bound: " << getName() << " - " << this <<
            ", " << __FILE__ << ", " << __LINE__ << std::endl;
    exit(1);
  }
  return sym->fullname();
}

std::string ScopedName::cppfullname(SymbolTable* forstab) const
{
  if (!sym) {
    std::cerr << "ERROR: Symbol not bound: " << getName() << " - " << this <<
            ", " << __FILE__ << ", " << __LINE__ << std::endl;
    exit(1);
  }
  return sym->cppfullname(forstab);
}

ScopedNameList::ScopedNameList()
{
}

ScopedNameList::~ScopedNameList()
{
    for (std::vector<ScopedName*>::iterator iter = list.begin();
        iter != list.end(); iter++) {
        delete *iter;
    }
}

void ScopedNameList::prepend(ScopedName* name)
{
    list.insert(list.begin(), name);
}

void ScopedNameList::add(ScopedName* name)
{
  list.push_back(name);
}

std::vector<ScopedName*> const& ScopedNameList::getList() const
{
  return list;
}

std::string ScopedNameList::fullsignature() const
{
    std::string s;
    if (list.empty()) {
        return s;
    }
    int c = 0;
    for (std::vector<ScopedName*>::const_iterator iter = list.begin();
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
    t = new BuiltinType(BuiltinType::COMPLEX_DOUBLE_NAME, "special");
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
    t = new BuiltinType(BuiltinType::COMPLEX_FLOAT_NAME, "special");
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

static std::map<std::pair<Type*, int>, Type*> arrays;

Type* Type::arraytype(Type* t, int dim)
{
  if (dim == 0) {
    dim = 1;
  }
  std::map<std::pair<Type*, int>, Type*>::iterator iter =
    arrays.find(std::pair<Type*, int>(t, dim));
  if (iter == arrays.end()) {
    return (arrays[std::pair<Type*, int>(t, dim)] = new ArrayType(t, dim));
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

BuiltinType::BuiltinType(const std::string& cname, const std::string& nexusname)
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
  if (nexusname == "special") { // string, complex<float>, complex<double>
    return "std::" + cname;
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

NamedType::NamedType(const std::string& curfile, int lineno, ScopedName* name)
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
  } else if (name->getSymbol()->getDefinition()->isEmitted()) {
    return name->cppfullname(localScope) + "::pointer";
  } else {
    return "CCALib::SmartPointer< " + name->cppfullname(localScope) + " >";
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
}

Enum::~Enum()
{
  for(std::vector<Enumerator*>::iterator iter = list.begin();
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
}

Enumerator::Enumerator(const std::string& curfile, int lineno,
		       const std::string& name, int value)
  : curfile(curfile), lineno(lineno), name(name), have_value(true),
    value(value)
{
}

Enumerator::~Enumerator()
{
}

VersionList::VersionList()
{
}

VersionList::~VersionList()
{
  for(std::vector<Version*>::iterator iter = list.begin();
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
  std::ostringstream o;
  o << v;
  version = o.str();
}

Version::Version(const std::string& id, const std::string& version)
  : id(id), version(version)
{
}

Version::~Version()
{
}

SpecificationList::SpecificationList()
{
}

SpecificationList::~SpecificationList()
{
  for(std::vector<Specification*>::iterator iter = specs.begin();
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
  for(std::vector<Specification*>::iterator iter = specs.begin();
      iter != specs.end(); iter++){
    if ((*iter)->isImport) {
      (*iter)->packages->processImports();
    }
  }
	        
  // "processImports not finished!" -- Might want to remove unnecessary packages here;
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
  for(std::vector<DistributionArray*>::iterator iter=list.begin();iter != list.end();iter++){
    delete *iter;
  }
}

void DistributionArrayList::add(DistributionArray* d)
{
  list.push_back(d);
}
