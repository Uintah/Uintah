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

#include <vector>
#include <string>
#include <iosfwd>

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
class Interface;
class EmitState;
class SState;

class Definition {
public:
    virtual ~Definition();
    virtual void staticCheck(SymbolTable*)=0;
    SymbolTable* getSymbolTable() const;
    std::string fullname() const;
    virtual void emit(EmitState& out)=0;
protected:
    Definition(const std::string& curfile, int lineno,
	       const std::string& name);
    std::string curfile;
    int lineno;
    std::string name;
    SymbolTable* symbols;
    mutable bool emitted_declaration;
    friend class Symbol;
    bool do_emit;

    void checkMethods(const std::vector<Method*>&);
};

class CI : public Definition {
public:
    CI(const std::string& curfile, int lineno, const std::string& name,
       MethodList*);
    virtual ~CI();
    void gatherParents(std::vector<CI*>& parents) const;
    void gatherParentInterfaces(std::vector<Interface*>& parents) const;
    void gatherMethods(std::vector<Method*>&) const;
    void gatherVtable(std::vector<Method*>&, bool onlyNewMethods) const;
    std::vector<Method*>& myMethods();
    std::string cppfullname(SymbolTable* forpackage) const;
    std::string cppclassname() const;
    MethodList* getMethods() const;
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
    std::vector<Interface*> parent_ifaces;
    MethodList* mymethods;
private:

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

    Argument(Mode mode, Type*);
    Argument(Mode mode, Type*, const std::string& id);
    ~Argument();

    void staticCheck(SymbolTable* names) const;
    bool matches(const Argument* arg, bool checkmode) const;
    std::string fullsignature() const;

    void emit_unmarshal(EmitState& e, const std::string& var,
			const std::string& qty,
			const std::string& bufname, bool declare) const;
    void emit_marshal(EmitState& e, const std::string& var,
		      const std::string& qty,
		      const std::string& bufname, bool top) const;
    void emit_declaration(EmitState& e, const std::string& var) const;
    void emit_marshalsize(EmitState& e, const std::string& var,
			  const std::string& sizevar,
			  const std::string& qty) const;
    void emit_prototype(SState&, SymbolTable* localScope) const;
    void emit_prototype_defin(SState&, const std::string&,
			      SymbolTable* localScope) const;
    Mode getMode() const;
private:
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
    void staticCheck(SymbolTable*) const;

    std::string fullsignature() const;

    std::vector<Argument*>& getList();
private:
    std::vector<Argument*> list;
};

class Class : public CI {
public:
    Class(const std::string& curfile, int lineno, const std::string& name,
	  ScopedName* class_extends, ScopedNameList* class_implements,
	  MethodList*);
    Class(const std::string& curfile, int lineno, const std::string& name);
    virtual ~Class();
    virtual void staticCheck(SymbolTable*);
    Class* getParentClass() const;
    Class* findParent(const std::string&);
    Method* findMethod(const Method*, bool recurse) const;
private:
    ScopedName* class_extends;
    ScopedNameList* class_implements;
};

class DefinitionList {
public:
    DefinitionList();
    ~DefinitionList();

    void add(Definition*);
    void staticCheck(SymbolTable*) const;
    void emit(EmitState& out);
private:
    std::vector<Definition*> list;
};

class Interface : public CI {
public:
    Interface(const std::string& curfile, int lineno, const std::string& id,
	      ScopedNameList* interface_extends, MethodList*);
    Interface(const std::string& curfile, int lineno, const std::string& id);
    virtual ~Interface();
    virtual void staticCheck(SymbolTable*);
    Method* findMethod(const Method*) const;
private:
    ScopedNameList* interface_extends;
};

class Method {
public:
    Method(const std::string& curfile, int lineno,
	   Type* return_type, const std::string& name, ArgumentList* args,
	   ScopedNameList* throws_clause);
    ~Method();

    enum Modifier {
	Abstract, Final, Static, Unknown
    };

    void setModifier(Modifier);
    Modifier getModifier() const;
    void staticCheck(SymbolTable* names);

    void setClass(Class* c);
    void setInterface(Interface* c);

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
protected:
    friend class CI;
    int handlerNum;
    int handlerOff;
private:
    std::string curfile;
    int lineno;
    Type* return_type;
    std::string name;
    ArgumentList* args;
    ScopedNameList* throws_clause;
    Modifier modifier;
    Class* myclass;
    Interface* myinterface;
    bool checked;
};

class MethodList {
public:
    MethodList();
    ~MethodList();

    void add(Method*);
    void staticCheck(SymbolTable*) const;
    void setClass(Class* c);
    void setInterface(Interface* c);
    Method* findMethod(const Method* match) const;
    void gatherMethods(std::vector<Method*>&) const;
    void gatherVtable(std::vector<Method*>&) const;
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
    virtual void emit(EmitState& out);
private:
    std::string name;
    DefinitionList* definition;
};

class ScopedName {
public:
    ScopedName();
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

class Specification {
public:
    Specification();
    ~Specification();

    void add(DefinitionList*);
    void staticCheck();
    void emit(std::ostream& out, std::ostream& headerout,
	      const std::string& hname) const;
private:
    std::vector<DefinitionList*> list;
    SymbolTable* globals;
};

class Type {
protected:
    Type();
public:
    virtual ~Type();
    virtual void staticCheck(SymbolTable* names) const=0;
    virtual void emit_unmarshal(EmitState& e, const std::string& arg,
				const std::string& qty,
				const std::string& bufname, bool declare) const=0;
    virtual void emit_marshal(EmitState& e, const std::string& arg,
			      const std::string& qty,
			      const std::string& bufname,
			      bool top) const=0;
    virtual void emit_declaration(EmitState& e, const std::string& var) const=0;
    virtual void emit_marshalsize(EmitState& e, const std::string& arg,
				  const std::string& sizevar,
				  const std::string& qty) const=0;
    virtual void emit_rettype(EmitState& e, const std::string& name) const=0;
    virtual bool array_use_pointer() const = 0;
    virtual bool uniformsize() const=0;

    enum ArgContext {
	ReturnType,
	ArgIn,
	ArgOut,
	ArgInOut,
	ArrayTemplate
    };
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
    static Type* stringtype();
    static Type* voidtype();

    static Type* arraytype(Type*, int number);

};

class BuiltinType : public Type {
public:
    virtual void staticCheck(SymbolTable* names) const;
    virtual void emit_unmarshal(EmitState& e, const std::string& arg,
				const std::string& qty,
				const std::string& bufname, bool declare) const;
    virtual void emit_marshal(EmitState& e, const std::string& arg,
			      const std::string& qty,
			      const std::string& bufname, bool top) const;
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
    std::string cname;
    std::string nexusname;
};

class NamedType : public Type {
public:
    NamedType(const std::string& curfile, int lineno, ScopedName*);
    virtual ~NamedType();
    virtual void staticCheck(SymbolTable* names) const;
    virtual void emit_unmarshal(EmitState& e, const std::string& arg,
				const std::string& qty,
				const std::string& bufname, bool declare) const;
    virtual void emit_marshal(EmitState& e, const std::string& arg,
			      const std::string& qty,
			      const std::string& bufname, bool top) const;
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
    std::string curfile;
    int lineno;
    ScopedName* name;
};

class ArrayType : public Type {
public:
    virtual void staticCheck(SymbolTable* names) const;
    virtual void emit_unmarshal(EmitState& e, const std::string& arg,
				const std::string& qty,
				const std::string& bufname, bool declare) const;
    virtual void emit_marshal(EmitState& e, const std::string& arg,
			      const std::string& qty,
			      const std::string& bufname, bool top) const;
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
    ArrayType(Type* subtype, int dim);
    virtual ~ArrayType();
private:
    Type* subtype;
    int dim;
};

#endif

