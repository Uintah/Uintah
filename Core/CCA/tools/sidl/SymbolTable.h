
#ifndef SYMBOLTABLE_H
#define SYMBOLTABLE_H

#include <map>
#include <string>

class Definition;
class Method;
class ScopedName;
class SymbolTable;
class EmitState;

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
