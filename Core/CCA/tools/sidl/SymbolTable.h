
#ifndef SYMBOLTABLE_H
#define SYMBOLTABLE_H

#include <map>
#include <string>

class Definition;
class Method;
class ScopedName;

class Symbol {
public:
    Symbol(const std::string& name);
    ~Symbol();

    enum Type {
	ClassType,
	PackageType,
	InterfaceType,
	MethodType
    };
    void setType(Type type);
    Type getType() const;
    const std::string& getName() const;
    Definition* getDefinition() const;
    void setDefinition(Definition*);
    Method* getMethod() const;
    void setMethod(Method*);
    std::string fullname() const;
protected:
    friend class SymbolTable;
    void setSymbolTable(SymbolTable*);
private:
    Type type;
    std::string name;
    Definition* definition;
    Method* method;
    SymbolTable* symtab;
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

    std::string fullname(const std::string& name) const;
    std::string fullname() const;
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
