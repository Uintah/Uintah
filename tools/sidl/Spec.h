
#ifndef SPEC_H
#define SPEC_H 1

#include <vector>
#include <string>

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

class Definition {
public:
    virtual ~Definition();
    virtual void staticCheck(SymbolTable*)=0;
    SymbolTable* getSymbolTable() const;
    std::string fullname() const;
protected:
    Definition(const std::string& curfile, int lineno,
	       const std::string& name);
    std::string curfile;
    int lineno;
    std::string name;
    SymbolTable* symbols;

    void checkMethods(const std::vector<Method*>&);
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
private:
    std::vector<Argument*> list;
};

class Class : public Definition {
public:
    Class(const std::string& curfile, int lineno, const std::string& name,
	  ScopedName* class_extends, ScopedNameList* class_implements,
	  MethodList*);
    Class(const std::string& curfile, int lineno, const std::string& name);
    virtual ~Class();
    virtual void staticCheck(SymbolTable*);
    MethodList* getMethods() const;
    Class* getParentClass() const;
    Method* findMethod(const Method*, bool recurse) const;
    void gatherMethods(std::vector<Method*>&) const;
    Class* findParent(const std::string&);
private:
    ScopedName* class_extends;
    ScopedNameList* class_implements;
    MethodList* methods;
    Class* parent_class;
    std::vector<Interface*> parent_interfaces;
};

class DefinitionList {
public:
    DefinitionList();
    ~DefinitionList();

    void add(Definition*);
    void staticCheck(SymbolTable*) const;
private:
    std::vector<Definition*> list;
};

class Interface : public Definition {
public:
    Interface(const std::string& curfile, int lineno, const std::string& id,
	      ScopedNameList* interface_extends, MethodList*);
    Interface(const std::string& curfile, int lineno, const std::string& id);
    virtual ~Interface();
    virtual void staticCheck(SymbolTable*);
    MethodList* getMethods() const;
    Method* findMethod(const Method*) const;
    void gatherMethods(std::vector<Method*>&) const;
private:
    ScopedNameList* interface_extends;
    MethodList* methods;
    std::vector<Interface*> parent_interfaces;
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
private:
    std::vector<Method*> list;
};

class Package : public Definition {
public:
    Package(const std::string& curfile, int lineno, const std::string& name,
	    DefinitionList* definition);
    virtual ~Package();
    virtual void staticCheck(SymbolTable*);
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

    void bind(Symbol*);
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
    void staticCheck(SymbolTable* names) const;
private:
    std::vector<DefinitionList*> list;
};

class Type {
protected:
    Type();
public:
    virtual ~Type();
    virtual void staticCheck(SymbolTable* names) const=0;

    virtual bool matches(const Type*) const=0;
    virtual std::string fullname() const=0;

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
    virtual bool matches(const Type*) const;
    virtual std::string fullname() const;
protected:
    friend class Type;
    BuiltinType(const std::string& cname);
    virtual ~BuiltinType();
private:
    std::string cname;
};

class NamedType : public Type {
public:
    NamedType(const std::string& curfile, int lineno, ScopedName*);
    virtual ~NamedType();
    virtual void staticCheck(SymbolTable* names) const;
    virtual bool matches(const Type*) const;
    virtual std::string fullname() const;
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
    virtual bool matches(const Type*) const;
    virtual std::string fullname() const;
protected:
    friend class Type;
    ArrayType(Type* subtype, int dim);
    virtual ~ArrayType();
private:
    Type* subtype;
    int dim;
};

#endif
//
// $Log$
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
