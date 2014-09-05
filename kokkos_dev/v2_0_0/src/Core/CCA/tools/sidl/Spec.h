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


#ifndef SPEC_H
#define SPEC_H 1

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <set>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

class Symbol;
class SymbolTable;
class Method;
class MethodList;
class ScopedName;
class ScopedNameList;
class Type;
class Argument;
class SymbolTable;
class Class;
class BaseInterface;
class EmitState;
class SState;
class ArrayType;

enum ArgContext {
  ReturnType,
  ArgIn,
  ArgOut,
  ArgInOut,
  ArrayTemplate
};

enum storageT {
  noneStorage,
  doStore,
  doRetreive
};

class Definition {
public:
  virtual ~Definition();
  virtual void staticCheck(SymbolTable*)=0;
  virtual void gatherSymbols(SymbolTable*)=0;
  SymbolTable* getSymbolTable() const;
  std::string fullname() const;
  virtual void emit(EmitState& out)=0;
  void setName(const std::string& name);
  bool isEmitted() { return emitted_declaration; }
  bool isImport;
  std::string curfile;
protected:
  Definition(const std::string& curfile, int lineno,
	     const std::string& name);
  int lineno;
  std::string name;
  SymbolTable* symbols;
  mutable bool emitted_declaration;
  friend class Symbol;
  bool do_emit;

  void checkMethods(const std::vector<Method*>&);
};

class DistributionArray : public Definition {
public:
  DistributionArray(const std::string& curfile, int lineno, const std::string& name, 
		    ArrayType* arr_t);
  virtual ~DistributionArray();
  ArrayType* getArrayType();
  std::string getName();
  void gatherSymbols(SymbolTable* );
  void staticCheck(SymbolTable* );
  void emit(EmitState& );
private:
  ArrayType *array_t;
  std::string name;
};

class DistributionArrayList {
public:
  DistributionArrayList();
  ~DistributionArrayList();

  void add(DistributionArray*);
  void staticCheck(SymbolTable*) const;
  void gatherSymbols(SymbolTable*) const;
  void emit(EmitState& out);
private:
  std::vector<DistributionArray*> list;
};

class CI : public Definition {
public:
  CI(const std::string& curfile, int lineno, const std::string& name,
     MethodList*, DistributionArrayList* );
  virtual ~CI();
  void detectRedistribution();
  void gatherParents(std::vector<CI*>& parents) const;
  void gatherParentInterfaces(std::vector<BaseInterface*>& parents) const;
  void gatherMethods(std::vector<Method*>&) const;
  void gatherVtable(std::vector<Method*>&, bool onlyNewMethods) const;
  std::vector<Method*>& myMethods();
  std::string cppfullname(SymbolTable* forpackage) const;
  std::string cppclassname() const;
  MethodList* getMethods() const;
  int exceptionID;
protected:
  virtual void emit(EmitState& out);
  void emit_typeinfo(EmitState& e);
  void emit_proxyclass(EmitState& e);
  void emit_handlers(EmitState& e);
  void emit_handler_table(EmitState& e);
  void emit_handler_table_body(EmitState& e, int&, bool top);
  void emit_interface(EmitState& e);
  void emit_proxy(EmitState& e);
  void emit_header(EmitState& e);
  Class* parentclass;
  std::vector<BaseInterface*> parent_ifaces;
  MethodList* mymethods;
  DistributionArrayList* mydistarrays;
private:
  bool doRedistribution; 
  int callerDistHandler;
  bool singly_inherited() const;
  void emit_recursive_vtable_comment(EmitState&, bool);
  bool iam_class();
  int isaHandler;
  int deleteReferenceHandler;
  int vtable_base;
};

class Argument {
public:
  enum Mode {
    In, Out, InOut
  };

  Argument(bool copy, Mode mode, Type*);
  Argument(bool copy, Mode mode, Type*, const std::string& id);
  ~Argument();

  void staticCheck(SymbolTable* names, Method* method) const;
  bool matches(const Argument* arg, bool checkmode) const;
  std::string fullsignature() const;

  int detectRedistribution();
  void emit_unmarshal(EmitState& e, const std::string& var,
		      const std::string& qty, const int handler,
		      ArgContext ctx, const bool specialRedis, bool declare) const;
  void emit_marshal(EmitState& e, const std::string& var, const std::string& qty, 
		    const int handler, bool top, ArgContext ctx, bool specialRedis, 
		    storageT bufferStore = noneStorage) const;
  void emit_declaration(EmitState& e, const std::string& var) const;
  void emit_marshalsize(EmitState& e, const std::string& var,
			const std::string& sizevar,
			const std::string& qty) const;
  void emit_prototype(SState&, SymbolTable* localScope) const;
  void emit_prototype_defin(SState&, const std::string&,
			    SymbolTable* localScope) const;
  Mode getMode() const;
  Type* getType() const;

private:
  bool copy;
  Mode mode;
  Type* type;
  std::string id;
};

class ArgumentList {
public:
  ArgumentList();
  ~ArgumentList();

  void prepend(Argument*);
  void add(Argument*);

  bool matches(const ArgumentList*, bool checkmode) const;
  int detectRedistribution();
  void staticCheck(SymbolTable*, Method*) const;

  std::string fullsignature() const;

  std::vector<Argument*>& getList();
private:
  std::vector<Argument*> list;
};

class Class : public CI {
public:
  enum Modifier {
    Abstract,
    None
  };
  Class(const std::string& curfile, int lineno, Modifier modifier,
	const std::string& name, ScopedName* class_extends,
	ScopedNameList* class_implementsall, ScopedNameList* class_implements,
	MethodList*, DistributionArrayList*);
  virtual ~Class();
  virtual void staticCheck(SymbolTable*);
  virtual void gatherSymbols(SymbolTable*);
  Class* getParentClass() const;
  Class* findParent(const std::string&);
  Method* findMethod(const Method*, bool recurse) const;
private:
  Modifier modifier;
  ScopedName* class_extends;
  ScopedNameList* class_implementsall;
  ScopedNameList* class_implements;
};

class DefinitionList {
public:
  DefinitionList();
  ~DefinitionList();

  void add(Definition*);
  void staticCheck(SymbolTable*) const;
  void gatherSymbols(SymbolTable*) const;
  void emit(EmitState& out);
  void processImports();
  std::vector<Definition*> list;
};

class BaseInterface : public CI {
public:
  BaseInterface(const std::string& curfile, int lineno, const std::string& id,
	    ScopedNameList* interface_extends, MethodList*, DistributionArrayList* );
  virtual ~BaseInterface();
  virtual void staticCheck(SymbolTable*);
  virtual void gatherSymbols(SymbolTable*);
  Method* findMethod(const Method*) const;
private:
  ScopedNameList* interface_extends;
};

class Method {
public:
  enum Modifier {
    Abstract, Final, Static, NoModifier, Unknown
  };

  enum Modifier2 {
    Local, Oneway, None
  };

  Method(const std::string& curfile, int lineno, bool copy_return,
	 Type* return_type, const std::string& name, ArgumentList* args,
	 Modifier2 modifier2,  ScopedNameList* throws_clause);
  ~Method();

  bool detectRedistribution();
  void setModifier(Modifier);
  Modifier getModifier() const;
  void staticCheck(SymbolTable* names);

  void setClass(Class* c);
  void setInterface(BaseInterface* c);

  enum MatchWhich {
    TypesOnly,
    TypesAndModes,
    TypesModesAndThrows,
    TypesModesThrowsAndReturnTypes
  };
  bool matches(const Method*, MatchWhich match) const;

  bool getChecked() const;
  const std::string& getCurfile() const;
  int getLineno() const;
  std::string fullname() const;
  std::string fullsignature() const;

  void emit_handler(EmitState& e, CI* emit_class) const;
  void emit_comment(EmitState& e, const std::string& leader, bool filename) const;
  enum Context {
    Normal,
    PureVirtual
  };
  void emit_prototype(SState& s,  Context ctx,
		      SymbolTable* localScope) const;
  void emit_prototype_defin(EmitState& e, const std::string& prefix,
			    SymbolTable* localScope) const;
  void emit_proxy(EmitState& e, const std::string& fn,
		  SymbolTable* localScope) const;

  bool reply_required() const;
  std::string get_classname() const;
  bool isCollective;
  int numRedisMessages;
protected:
  friend class CI;
  int handlerNum;
  int handlerOff;
private:
  bool doRedistribution;
  std::string curfile;
  int lineno;
  bool copy_return;
  Type* return_type;
  std::string name;
  ArgumentList* args;
  Modifier2 modifier2;
  ScopedNameList* throws_clause;
  Modifier modifier;
  Class* myclass;
  BaseInterface* myinterface;
  bool checked;
};

class MethodList {
public:
  MethodList();
  ~MethodList();

  void add(Method*);
  void staticCheck(SymbolTable*) const;
  void setClass(Class* c);
  void setInterface(BaseInterface* c);
  Method* findMethod(const Method* match) const;
  void gatherMethods(std::vector<Method*>&) const;
  void gatherVtable(std::vector<Method*>&) const;
  bool detectRedistribution();
  std::vector<Method*>& getList();
private:
  std::vector<Method*> list;
};

class Package : public Definition {
public:
  Package(const std::string& curfile, int lineno, const std::string& name,
	  DefinitionList* definition);
  virtual ~Package();
  virtual void staticCheck(SymbolTable*);
  virtual void gatherSymbols(SymbolTable*);
  virtual void emit(EmitState& out);
  DefinitionList* definition;
};

class Enumerator {
public:
  Enumerator(const std::string& curfile, int lineno, const std::string& name);
  Enumerator(const std::string& curfile, int lineno, const std::string& name,
	     int value);
  ~Enumerator();

  void assignValues1(SymbolTable*, std::set<int>& occupied);
  void assignValues2(SymbolTable*, std::set<int>& occupied, int& nextvalue);
  void gatherSymbols(SymbolTable*);
  void emit(EmitState& out, bool first);
public:
  std::string curfile;
  int lineno;
  std::string name;
  bool have_value;
  int value;
};

class Enum : public Definition {
public:
  Enum(const std::string& curfile, int lineno, const std::string& name);
  virtual ~Enum();
  void add(Enumerator* e);
  virtual void staticCheck(SymbolTable*);
  virtual void gatherSymbols(SymbolTable*);
  virtual void emit(EmitState& out);
private:
  std::vector<Enumerator*> list;
};

class ScopedName {
public:
  ScopedName();
  ScopedName(const std::string&);
  ScopedName(const std::string&, const std::string&);
  ~ScopedName();

  void prepend(const std::string&);
  void add(const std::string&);
  void set_leading_dot(bool);
  bool getLeadingDot() const;

  std::string getName() const;

  int nnames() const;
  const std::string& name(int i) const;

  bool matches(const ScopedName*) const;
  std::string fullname() const;
  std::string cppfullname(SymbolTable* forpackage=0) const;

  void bind(Symbol*);
  Symbol* getSymbol() const;
private:
  std::vector<std::string> names;
  bool leading_dot;
  Symbol* sym;
};

class ScopedNameList {
public:
  ScopedNameList();
  ~ScopedNameList();

  void prepend(ScopedName*);
  void add(ScopedName*);
  const std::vector<ScopedName*>& getList() const;

  bool matches(const ScopedNameList*) const;
  std::string fullsignature() const;
private:
  std::vector<ScopedName*> list;
};

class Version {
public:
  Version(const std::string& id, int version);
  Version(const std::string& id, const std::string& version);
  ~Version();
private:
  std::string id;
  std::string version;
};

class VersionList {
public:
  VersionList();
  ~VersionList();
  void add(Version*);
private:
  std::vector<Version*> list;
};

class Specification {
public:
  Specification(VersionList* versions, ScopedNameList* imports,
		DefinitionList* packages);
  ~Specification();

  void setTopLevel();
  void staticCheck(SymbolTable* globals);
  void gatherSymbols(SymbolTable* globals);
  void processImports();
  bool isImport;
  VersionList* versions;
  ScopedNameList* imports;
  DefinitionList* packages;
  bool toplevel;
};

class SpecificationList {
public:
  SpecificationList();
  ~SpecificationList();
  void add(Specification* spec);
  void processImports();
  void staticCheck();
  void emit(std::ostream& out, std::ostream& headerout,
	    const std::string& hname) const;
public:
  SymbolTable* globals;
  std::vector<Specification*> specs;
};

class Type {
protected:
  Type();
public:
  virtual ~Type();

  virtual int  detectRedistribution() = 0;
  virtual void staticCheck(SymbolTable* names, Method* method) = 0;
  virtual void emit_unmarshal(EmitState& e, const std::string& arg,
			      const std::string& qty, const int handler, ArgContext ctx,
			      const bool specialRedis, bool declare) const=0;
  virtual void emit_marshal(EmitState& e, const std::string& arg, const std::string& qty,
			    const int handler, bool top, ArgContext ctx, bool specialRedis, 
			    storageT bufferStore = noneStorage) const=0;
  virtual void emit_declaration(EmitState& e, const std::string& var) const=0;
  virtual void emit_marshalsize(EmitState& e, const std::string& arg,
				const std::string& sizevar,
				const std::string& qty) const=0;
  virtual void emit_rettype(EmitState& e, const std::string& name) const=0;
  virtual bool array_use_pointer() const = 0;
  virtual bool uniformsize() const=0;

  virtual void emit_prototype(SState& s, ArgContext ctx,
			      SymbolTable* localScope) const=0;

  virtual bool matches(const Type*) const=0;
  virtual std::string fullname() const=0;
  virtual std::string cppfullname(SymbolTable* localScope) const=0;

  virtual bool isvoid() const=0;

  static Type* arraytype();
  static Type* booltype();
  static Type* chartype();
  static Type* dcomplextype();
  static Type* doubletype();
  static Type* fcomplextype();
  static Type* floattype();
  static Type* inttype();
  static Type* longtype();
  static Type* opaquetype();
  static Type* stringtype();
  static Type* voidtype();

  static Type* arraytype(Type*, int number);

};

class BuiltinType : public Type {
public:
  virtual void staticCheck(SymbolTable* names, Method* method);
  int detectRedistribution();
  virtual void emit_unmarshal(EmitState& e, const std::string& arg,
			      const std::string& qty, const int handler, ArgContext ctx,
			      const bool specialRedis, bool declare) const;
  virtual void emit_marshal(EmitState& e, const std::string& arg, const std::string& qty, 
			    const int handler, bool top, ArgContext ctx, bool specialRedis, 
			    storageT bufferStore = noneStorage) const;
  virtual void emit_declaration(EmitState& e, const std::string& var) const;
  virtual void emit_marshalsize(EmitState& e, const std::string& arg,
				const std::string& sizevar,
				const std::string& qty) const;
  virtual void emit_rettype(EmitState& e, const std::string& name) const;
  virtual void emit_prototype(SState& s, ArgContext ctx,
			      SymbolTable* localScope) const;
  virtual bool array_use_pointer() const ;
  virtual bool uniformsize() const;
  virtual bool matches(const Type*) const;
  virtual std::string fullname() const;
  virtual std::string cppfullname(SymbolTable* localScope) const;
  virtual bool isvoid() const;
protected:
  friend class Type;
  BuiltinType(const std::string& cname, const std::string& nexusname);
  virtual ~BuiltinType();
private:
  Method* thisMethod;
  std::string cname;
  std::string nexusname;
};

class NamedType : public Type {
public:
  NamedType(const std::string& curfile, int lineno, ScopedName*);
  virtual ~NamedType();
  virtual void staticCheck(SymbolTable* names, Method* method);
  int detectRedistribution();
  virtual void emit_unmarshal(EmitState& e, const std::string& arg,
			      const std::string& qty, const int handler, ArgContext ctx,
			      const bool specialRedis, bool declare) const;
  virtual void emit_marshal(EmitState& e, const std::string& arg, const std::string& qty, 
			    const int handler, bool top, ArgContext ctx, bool specialRedis, 
			    storageT bufferStore = noneStorage) const;
  virtual void emit_declaration(EmitState& e, const std::string& var) const;
  virtual void emit_marshalsize(EmitState& e, const std::string& arg,
				const std::string& sizevar,
				const std::string& qty) const;
  virtual void emit_rettype(EmitState& e, const std::string& name) const;
  virtual void emit_prototype(SState& s, ArgContext ctx,
			      SymbolTable* localScope) const;
  virtual bool array_use_pointer() const;
  virtual bool uniformsize() const;
  virtual bool matches(const Type*) const;
  virtual std::string fullname() const;
  virtual std::string cppfullname(SymbolTable* localScope) const;
  virtual bool isvoid() const;
protected:
  friend class Type;
private:
  Method* thisMethod;
  std::string curfile;
  int lineno;
  ScopedName* name;
};

class ArrayType : public Type {
public:
  virtual void staticCheck(SymbolTable* names, Method* method);
  int detectRedistribution();
  virtual void emit_unmarshal(EmitState& e, const std::string& arg,
			      const std::string& qty, const int handler, ArgContext ctx,
			      const bool specialRedis, bool declare) const;
  virtual void emit_marshal(EmitState& e, const std::string& arg, const std::string& qty, 
			    const int handler, bool top, ArgContext ctx, bool specialRedis, 
			    storageT bufferStore = noneStorage) const;
  virtual void emit_declaration(EmitState& e, const std::string& var) const;
  virtual void emit_marshalsize(EmitState& e, const std::string& arg,
				const std::string& sizevar,
				const std::string& qty) const;
  virtual void emit_rettype(EmitState& e, const std::string& name) const;
  virtual void emit_prototype(SState& s, ArgContext ctx,
			      SymbolTable* localScope) const;
  virtual bool array_use_pointer() const;
  virtual bool uniformsize() const;
  virtual bool matches(const Type*) const;
  virtual std::string fullname() const;
  virtual std::string cppfullname(SymbolTable* localScope) const;
  virtual bool isvoid() const;
  ArrayType(Type* subtype, int dim);
  virtual ~ArrayType();
  int dim;
  Type* subtype;
protected:
  friend class Type;
private:
  Method* thisMethod;
};

#endif



