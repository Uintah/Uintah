
#include "Spec.h"
#include "SymbolTable.h"
#include <iostream>

using std::vector;
using std::cerr;
using std::map;

void Specification::staticCheck(SymbolTable* names) const
{
    for(vector<DefinitionList*>::const_iterator iter=list.begin();iter != list.end(); iter++){
	(*iter)->staticCheck(names);
    }
}

void DefinitionList::staticCheck(SymbolTable* names) const
{
    for(vector<Definition*>::const_iterator iter=list.begin();iter != list.end(); iter++){
	(*iter)->staticCheck(names);
    }
}

void Package::staticCheck(SymbolTable* names)
{
    Symbol* n=names->lookup(name, false);
    if(!n){
	n=new Symbol(name);
	names->insert(n);
	symbols=new SymbolTable(names, name);
	n->setType(Symbol::PackageType);
	n->setDefinition(this);
    } else {
	symbols=n->getDefinition()->getSymbolTable();
    }
    definition->staticCheck(symbols);
}

void Interface::staticCheck(SymbolTable* names)
{
    Symbol* n=names->lookup(name, false);
    if(n){
	switch(n->getType()){
	case Symbol::ClassType:
	case Symbol::PackageType:
	case Symbol::MethodType:
	    cerr << curfile << ':' << lineno 
		 << ": (100) Re-definition of " << names->fullname(name) << " as interface\n";
	    exit(1);
	    break;
	case Symbol::InterfaceType:
	    if(((Interface*)n->getDefinition())->methods){
		cerr << curfile << ':' << lineno
		     << ": (101) Re-definition of interface " << names->fullname(name) << '\n';
		exit(1);
	    }
	    break;
	}
    }

    if(!n){
	n=new Symbol(name);
	names->insert(n);
	symbols=new SymbolTable(names, name);
    } else {
	symbols=n->getDefinition()->getSymbolTable();
    }
    n->setType(Symbol::InterfaceType);
    n->setDefinition(this);

    /* Handle builtin types - Object  */
    if(methods && !interface_extends && fullname() != ".CIA.Interface"){
	if(!interface_extends)
	    interface_extends=new ScopedNameList();
	interface_extends->prepend(new ScopedName("CIA", "Interface"));
    }

    /* Check extends list */
    if(interface_extends){
	const vector<ScopedName*>& list=interface_extends->getList();
	for(vector<ScopedName*>::const_iterator iter=list.begin();iter != list.end();iter++){
	    Symbol* s=names->lookup(*iter);
	    if(!s){
		cerr << curfile << ':' << lineno 
		     << ": (102) Interface extends unknown type: " 
		     << (*iter)->getName() << '\n';
		exit(1);
	    }
	    if(s->getType() != Symbol::InterfaceType){
		cerr << curfile << ':' << lineno
		     << ": (103) Interface extends a non-interface: " 
		     << (*iter)->getName() << '\n';;
		exit(1);
	    }
	    (*iter)->bind(s);
	    Definition* d=s->getDefinition();
	    Interface* i=(Interface*)d;
	    if(!i->getMethods()){
		cerr << curfile << ':' << lineno
		     << ": (104) Interface extends an interface with no declaration (forward declaration only): " 
		     << (*iter)->getName() << '\n';
		exit(1);
	    }
	    parent_interfaces.push_back(i);
	}
    }

    /* Check methods */
    if(methods){
	methods->setInterface(this);
	methods->staticCheck(symbols);
	vector<Method*> allmethods;
	gatherMethods(allmethods);
	checkMethods(allmethods);
    }
}

void Class::staticCheck(SymbolTable* names)
{
    Symbol* n=names->lookup(name, false);
    if(n){
	switch(n->getType()){
	case Symbol::InterfaceType:
	case Symbol::PackageType:
	case Symbol::MethodType:
	    cerr << curfile << ':' << lineno 
		 << ": (105) Re-definition of " << names->fullname(name) << " as class\n";
	    exit(1);
	    break;
	case Symbol::ClassType:
	    if(((Class*)n->getDefinition())->methods){
		cerr << curfile << ':' << lineno
		     << ": (106) Re-definition of class " << names->fullname(name) << '\n';
		exit(1);
		break;
	    }
	}
    }

    if(!n){
	n=new Symbol(name);
	names->insert(n);
	symbols=new SymbolTable(names, name);
    } else {
	symbols=n->getDefinition()->getSymbolTable();
    }
    n->setType(Symbol::ClassType);
    n->setDefinition(this);

    /* Handle builtin types - Object  */
    if(methods && !class_extends && fullname() != ".CIA.Object"){
	class_extends=new ScopedName("CIA", "Object");
    }

    /* Check extends class */
    if(class_extends){
	Symbol* s=names->lookup(class_extends);
	if(!s){
	    cerr << curfile << ':' << lineno 
		 << ": (107) Class extends unknown class: " 
		 << class_extends->getName() << '\n';;
	    exit(1);
	}
	if(s->getType() != Symbol::ClassType){
	    cerr << curfile << ':' << lineno
		 << ": (108) Class extends a non-class: " 
		 << class_extends->getName() << '\n';;
	    exit(1);
	}
	class_extends->bind(s);
	Definition* d=s->getDefinition();
	Class* c=(Class*)d;
	if(!c->methods){
	    cerr << curfile << ':' << lineno
		 << ": (109) Class extends a class with no declaration (forward declaration only): " << class_extends->getName() << '\n';
	    exit(1);
	}
	parent_class=c;
    }
    /* Check implements list */
    if(class_implements){
	const vector<ScopedName*>& list=class_implements->getList();
	for(vector<ScopedName*>::const_iterator iter=list.begin();iter != list.end();iter++){
	    Symbol* s=names->lookup(*iter);
	    if(!s){
		cerr << curfile << ':' << lineno 
		     << ": (110) Class implements unknown interface: " 
		     << (*iter)->getName() << '\n';;
		exit(1);
	    }
	    (*iter)->bind(s);
	    if(s->getType() != Symbol::InterfaceType){
		cerr << curfile << ':' << lineno
		     << ": (111) Class implements a non-interface: " 
		     << (*iter)->getName() << '\n';;
		exit(1);
	    }
	    Definition* d=s->getDefinition();
	    Interface* i=(Interface*)d;
	    if(!i->getMethods()){
		cerr << curfile << ':' << lineno
		     << ": (112) Class implements an interface with no declaration (forward declaration only): " 
		     << (*iter)->getName() << '\n';
		exit(1);
	    }
	    parent_interfaces.push_back(i);
	}
    }

    /* Check methods */
    if(methods){
	methods->setClass(this);
	methods->staticCheck(symbols);
	vector<Method*> allmethods;
	gatherMethods(allmethods);
	checkMethods(allmethods);
    }
}

Method* Class::findMethod(const Method* match, bool recurse) const
{
    Method* m=methods->findMethod(match);
    if(!m && recurse){
	if(parent_class)
	    return parent_class->findMethod(match, recurse);
	else
	    return 0;
    } else
	return m;
}

Method* Interface::findMethod(const Method* match) const
{
    Method* m=methods->findMethod(match);
    return m;
}

Method* MethodList::findMethod(const Method* match) const
{
    for(vector<Method*>::const_iterator iter=list.begin();iter != list.end();iter++){
	Method * m=*iter;
	if(m != match && m->getChecked()){
	    if(m->matches(match, Method::TypesOnly)){
		return m;
	    }
	}
    }
    return 0;
}

void MethodList::staticCheck(SymbolTable* names) const
{
    for(vector<Method*>::const_iterator iter=list.begin();iter != list.end();iter++){
	(*iter)->staticCheck(names);
    }
}

void Definition::checkMethods(const vector<Method*>& allmethods)
{
    for(vector<Method*>::const_iterator iter1=allmethods.begin();
	iter1 != allmethods.end();iter1++){
	for(vector<Method*>::const_iterator iter2=allmethods.begin();
	    iter2 != iter1;iter2++){
	    Method* m1=*iter1;
	    Method* m2=*iter2;
	    if(m1->matches(m2, Method::TypesOnly)){
		if(!m1->matches(m2, Method::TypesAndModes)){
		    cerr << m1->getCurfile() << ':' << m1->getLineno()
			 << ": (123) Method (" << m1->fullname() << ") does not match modes of related method (" << m2->fullname() << ") defined at " << m2->getCurfile() << ':' << m2->getLineno() << "\n";
		    exit(1);
		}
		if(!m1->matches(m2, Method::TypesModesAndThrows)){
		    cerr << m1->getCurfile() << ':' << m1->getLineno()
			 << ": (124) Method (" << m1->fullname() << ") does not match throws clause of related method (" << m2->fullname() << ") defined at " << m2->getCurfile() << ':' << m2->getLineno() << "\n";
		    exit(1);
		}
		if(!m1->matches(m2, Method::TypesModesThrowsAndReturnTypes)){
		    m1->matches(m2, Method::TypesModesThrowsAndReturnTypes);
		    cerr << m1->getCurfile() << ':' << m1->getLineno()
			 << ": (125) Method (" << m1->fullname() << ") does not match return type of related method (" << m2->fullname() << ") defined at " << m2->getCurfile() << ':' << m2->getLineno() << "\n";
		    cerr << m1->fullsignature() << "\n";
		    cerr << m2->fullsignature() << "\n";
		    exit(1);
		}
	    }
	}
    }
}

void Method::staticCheck(SymbolTable* names)
{
    /* Method shouldn't match any other in the local namespace */
    Symbol* n=names->lookup(name, false);
    if(n){
	/* See if types match */
	switch(n->getType()){
	case Symbol::InterfaceType:
	case Symbol::PackageType:
	case Symbol::ClassType:
	    cerr << curfile << ':' << lineno
		 << ": (113) Internal error - Re-definition of " << names->fullname(name) << " as a method\n";
	    exit(1);
	    break;
	case Symbol::MethodType:
	    /* This is handled below... */
	    break;
	}
    }

    /* Check return type and arguments */
    return_type->staticCheck(names);
    args->staticCheck(names);
    checked=true;

    /* Method shouldn't match another method exactly */
    if(myclass){
	Method* meth=myclass->findMethod(this, false);
	if(meth){
	    cerr << curfile << ':' << lineno
		 << ": (114) Re-definition of method " << names->fullname(name) << '\n';
	    exit(1);
	}
    }
    if(myinterface){
	Method* meth=myinterface->findMethod(this);
	if(meth){
	    cerr << curfile << ':' << lineno
		 << ": (115) Re-definition of method " << names->fullname(name) << '\n';
	    exit(1);
	}
    }

    /* Shouldn't override a final method */
    if(myclass){
	Class* parent=myclass->getParentClass();
	if(parent){
	    Method* m=parent->findMethod(this, true);
	    if(m){
		switch(m->getModifier()){
		case Method::Abstract:
		case Method::Static:
		case Method::Unknown:
		    // No problem...
		    break;
		case Method::Final:
		    cerr << curfile << ':' << lineno
			 << ": (117) Method attempts to override a final method: " << name << '\n';
		    exit(1);
		}
	    }
	}
    }

    /* Check throws clause */
    if(throws_clause){
	const vector<ScopedName*>& list=throws_clause->getList();
	map<Class*, int> thrown;
	for(vector<ScopedName*>::const_iterator iter=list.begin();iter != list.end();iter++){
	    Symbol* s=names->lookup(*iter);
	    if(!s){
		cerr << curfile << ':' << lineno
		     << ": (118) Method throws unknown type: "
		     << (*iter)->getName() << '\n';
		exit(1);
	    }
	    (*iter)->bind(s);
	    switch(s->getType()){
	    case Symbol::ClassType:
		{
		    Class* c=(Class*)s->getDefinition();
		    map<Class*, int>::iterator citer=thrown.find(c);
		    if(citer != thrown.end()){
			cerr << curfile << ':' << lineno
			     << ": (119) Method specifies redundant throw: "
			     << (*iter)->getName() << '\n';
			exit(1);
		    }
		    if(!c->getMethods()){
			cerr << curfile << ':' << lineno
			     << ": (126) method throws incomplete class: "
			     << (*iter)->getName() << '\n';
			exit(1);
		    }
		    Class* t=c->findParent(".CIA.Throwable");
		    if(!t){
			cerr << curfile << ':' << lineno
			     << ": (127) method must throw a derivative of .CIA.Throwable: "
			     << (*iter)->getName() << '\n';
			exit(1);
		    }
		    thrown[c]=1; // Just a dummy value
		}
		break;
	    case Symbol::MethodType:
	    case Symbol::PackageType:
	    case Symbol::InterfaceType:
		cerr << curfile << ':' << lineno
		     << ": (120) Method throws a non-class: "
		     << (*iter)->getName() << '\n';
		exit(1);
		break;
	    }
	}
    }

    n=new Symbol(name);
    names->insert(n);
    n->setType(Symbol::MethodType);
    n->setMethod(this);
}

bool Method::matches(const Method* m, Method::MatchWhich match) const
{
    if(name != m->name)
	return false;
    switch(match){
    case TypesOnly:
	return args->matches(m->args, false);
	break;
    case TypesAndModes:
	return args->matches(m->args, true);
	break;
    case TypesModesAndThrows:
	if((throws_clause && !m->throws_clause) ||
	   (!throws_clause && m->throws_clause))
	    return false;
	else
	    return args->matches(m->args, true)
		&& (!throws_clause || throws_clause->matches(m->throws_clause));
	break;
    case TypesModesThrowsAndReturnTypes:
	if((throws_clause && !m->throws_clause) ||
	   (!throws_clause && m->throws_clause))
	    return false;
	else
	    return args->matches(m->args, true)
		&& (!throws_clause || throws_clause->matches(m->throws_clause))
		&& return_type->matches(m->return_type);
	break;
    }
    cerr << "Error - should not reach this point!\n";
    exit(1);
    return false;
}

void ArgumentList::staticCheck(SymbolTable* names) const
{
    for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	(*iter)->staticCheck(names);
    }
}

void Argument::staticCheck(SymbolTable* names) const
{
    type->staticCheck(names);
}

bool ArgumentList::matches(const ArgumentList* a, bool checkmode) const
{
    if(list.size() != a->list.size())
	return false;
    for(vector<Argument*>::const_iterator iter1=list.begin(), iter2=a->list.begin();
	iter1 != list.end() && iter1 != a->list.end();
	iter1++, iter2++){
	if(!(*iter1)->matches(*iter2, checkmode))
	    return false;
    }
    return true;
}

bool Argument::matches(const Argument* a, bool checkmode) const
{
    if(checkmode){
	if(mode != a->mode)
	    return false;
    }
    return type->matches(a->type);
}

void BuiltinType::staticCheck(SymbolTable* names) const
{
    // Nothing to be done...
}

void NamedType::staticCheck(SymbolTable* names) const
{
    Symbol* s=names->lookup(name);
    if(!s){
	cerr << curfile << ':' << lineno
	     << ": (121) Unknown type in method declaration: "
	     << name->getName() << '\n';
	exit(1);
    }
    name->bind(s);

    switch(s->getType()){
    case Symbol::ClassType:
    case Symbol::InterfaceType:
	/* Ok... */
	break;
    case Symbol::PackageType:
    case Symbol::MethodType:
	cerr << curfile << ':' << lineno
	     << ": (122) Argument in method declaration is not a type name: "
	     << name->getName() << '\n';
	exit(1);
    }
}

void ArrayType::staticCheck(SymbolTable* names) const
{
    subtype->staticCheck(names);
}

bool ScopedNameList::matches(const ScopedNameList* l) const
{
    if(list.size() != l->list.size())
	return false;
    for(vector<ScopedName*>::const_iterator iter1=list.begin(), iter2=l->list.begin();
	iter1 != list.end() && iter2 != l->list.end();
	iter1++, iter2++){
	if(!(*iter1)->matches(*iter2))
	    return false;
    }
    return true;
}

bool ScopedName::matches(const ScopedName* n) const
{
    return fullname() == n->fullname();
}

bool BuiltinType::matches(const Type* t) const
{
    const BuiltinType* b=dynamic_cast<const BuiltinType*>(t);
    if(b){
	if(b->cname == cname)
	    return true;
    }
    return false;
}

bool ArrayType::matches(const Type* tt) const
{
    const ArrayType* a=dynamic_cast<const ArrayType*>(tt);
    if(a){
	if(subtype->matches(a->subtype) && dim == a->dim)
	    return true;
    }
    return false;
}

bool NamedType::matches(const Type* t) const
{
    const NamedType* n=dynamic_cast<const NamedType*>(t);
    if(n)
	return name->matches(n->name);
    else 
	return false;
}

void ScopedName::bind(Symbol* s)
{
    sym=s;
}
//
// $Log$
// Revision 1.2  1999/08/30 17:39:55  sparker
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
