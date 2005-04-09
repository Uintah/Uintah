
/*
 *  Args.h: Interface to Argument definition classes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_Classlib_Args_h
#define SCI_Classlib_Args_h 1

#include <Classlib/HashTable.h>
#include <Classlib/String.h>

// The base class for the arguments
class Arg_base {
    friend class ArgProcessor;
    clString usage_string;
    virtual int handle(int, char**, int&)=0;
    void usage();
    virtual void usage_specific()=0;
public:
    Arg_base(const clString& name, const clString& usage_string);
    virtual ~Arg_base();
};

class Arg_set;

// An item in an Arg_set
class Arg_item {
protected:
    Arg_set* itemset;
    int set;
    friend class Arg_set;
    friend class Arg_itemalias;
public:
    Arg_item(Arg_set*, const clString& name);
    ~Arg_item();
    int is_set();
};

// An alias for an item in an Arg_ste
class Arg_itemalias {
public:
    Arg_itemalias(Arg_item*, const clString& name);
};

// Base class for the exclusive/nonexclusive classes
class Arg_set : public Arg_base {
    int exclusive;
    int set;
    Arg_item* current_choice;
    HashTable<clString, Arg_item*> valid_choices;
    virtual int handle(int, char**, int&);
    virtual void usage_specific();
    void print_list();
public:
    Arg_set(const clString& name, const clString& usage_string,
	    int exclusive);
    virtual ~Arg_set();
    void register_item(const clString&, Arg_item*);
};

// Provide exclusive behavior
class Arg_exclusiveset : public Arg_set {
public:
    Arg_exclusiveset(const clString& name, const clString& usage_string);
    virtual ~Arg_exclusiveset();
};

// Provide non-exclusive behavior
class Arg_nonexclusiveset : public Arg_set {
public:
    Arg_nonexclusiveset(const clString& name, const clString& usage_string);
    virtual ~Arg_nonexclusiveset();
};

// An argument followed by a string
class Arg_stringval : public Arg_base {
    clString val;
    int set;
    virtual int handle(int, char**, int&);
    virtual void usage_specific();
public:
    Arg_stringval(const clString& name, const clString& value,
		  const clString& usage_string);
    virtual ~Arg_stringval();
    int is_set();
    clString value();
};

// An argument followed by an integer
class Arg_intval : public Arg_base {
    int val;
    int set;
    virtual int handle(int, char**, int&);
    virtual void usage_specific();
public:
    Arg_intval(const clString& name, int value,
	       const clString& usage_string);
    virtual ~Arg_intval();
    int is_set();
    int value();
};

// An argument followed by a double
class Arg_doubleval : public Arg_base {
    double val;
    int set;
    virtual int handle(int, char**, int&);
    virtual void usage_specific();
public:
    Arg_doubleval(const clString& name, double value,
		  const clString& usage_sting);
    virtual ~Arg_doubleval();
    int is_set();
    double value();
};

// A simple flag
class Arg_flag : public Arg_base {
    int set;
    virtual int handle(int, char**, int&);
    virtual void usage_specific();
public:
    Arg_flag(const clString& name, const clString& usage_string);
    virtual ~Arg_flag();
    int is_set();
};

// Add another name for any Arg
class Arg_alias {
public:
    Arg_alias(Arg_base*, const clString& name);
};

#endif
