
/*
 *  Args.cc: Implementation of Argument definition classes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Args.h>
#include <Classlib/ArgProcessor.h>
#include <Classlib/AVLTree.h>
#include <iostream.h>

// Implementation of the Arg_base class
Arg_base::Arg_base(const clString& name, const clString& usage_string)
: usage_string(usage_string)
{
    ArgProcessor::register_arg(name, this);
}

Arg_base::~Arg_base()
{
}

void Arg_base::usage()
{
    usage_specific();
    cerr << usage_string;
}

// Implementation of the Arg_alias class
Arg_alias::Arg_alias(Arg_base* ab, const clString& name)
{
    ArgProcessor::register_arg(name, ab);
}

// Implementation of the Arg_item class
Arg_item::Arg_item(Arg_set* itemset, const clString& name)
: itemset(itemset), set(0)
{
    itemset->register_item(name, this);
}

Arg_item::~Arg_item()
{
}

int Arg_item::is_set()
{
    return set;
}

// Implementation of the Arg_itemalias class
Arg_itemalias::Arg_itemalias(Arg_item* item, const clString& name)
{
    item->itemset->register_item(name, item);
}

// Implementation of the Arg_set class
Arg_set::Arg_set(const clString& name, const clString& usage_string,
		 int exclusive)
: Arg_base(name, usage_string), exclusive(exclusive), current_choice(0), set(0)
{
}

Arg_set::~Arg_set()
{
}

void Arg_set::register_item(const clString& name, Arg_item* item)
{
    valid_choices.insert(name, item);
    if(exclusive && !current_choice){
	// The default value is the first one...
	current_choice=item;
	item->set=0;
    }
}

void Arg_set::usage_specific()
{
    cerr << "[one of: ";
    print_list();
    cerr << "]" << endl << "\t\t\t";
}

void Arg_set::print_list()
{
    // Alphabetize them...
    AVLTree<clString, int> tree;
    HashTableIter<clString, Arg_item*> hiter(&valid_choices);
    for(hiter.first();hiter.ok();++hiter)
	tree.insert(hiter.get_key(), 0);
    AVLTreeIter<clString, int> iter(&tree);
    for(iter.first();iter.ok();++iter){
	cerr << iter.get_key();
	if(iter.ok()){
	    // Have next...
	    cerr << ", ";
	}
    }
}

int Arg_set::handle(int argc, char** argv, int& idx)
{
    if(idx >= argc){
	cerr << argv[idx-1] << "must be followed by one of ";
	print_list();
	cerr << endl;
	return 0; // No argument
    }
    clString next(argv[idx++]);
    Arg_item* item;
    if(!valid_choices.lookup(next, item)){
	cerr << argv[idx-1] << " is not a valid option for " << argv[idx-2] << endl;
	cerr << "valid options are: ";
	print_list();
	cerr << endl;
	return 0; // Invalid option
    }
    if(exclusive){
	if(current_choice)
	    current_choice->set=1;
	current_choice=item;
    }
    item->set=1;
    set=1;
    return 1; // Everything ok
}

// Implementation of the Arg_exclusiveset class
Arg_exclusiveset::Arg_exclusiveset(const clString& name,
				   const clString& usage_string)
: Arg_set(name, usage_string, 1)
{
}

Arg_exclusiveset::~Arg_exclusiveset()
{
}

// Implementation of the Arg_nonexclusiveset class
Arg_nonexclusiveset::Arg_nonexclusiveset(const clString& name,
					 const clString& usage_string)
: Arg_set(name, usage_string, 0)
{
}

Arg_nonexclusiveset::~Arg_nonexclusiveset()
{
}

// Implementation of the Arg_stringval class
Arg_stringval::Arg_stringval(const clString& name, const clString& val,
			     const clString& usage_string)
: Arg_base(name, usage_string), val(val), set(0)
{
}

Arg_stringval::~Arg_stringval()
{
}

int Arg_stringval::is_set()
{
    return set;
}

clString Arg_stringval::value()
{
    return val;
}

int Arg_stringval::handle(int argc, char** argv, int& idx)
{
    if(idx >= argc){
	cerr << argv[idx-1] << " must be followed by a string parameter" << endl;
	return 0; // No argument
    }
    clString next(argv[idx++]);
    val=next;
    set=1;
}

void Arg_stringval::usage_specific()
{
    cerr << "[string argument]" << endl << "\t\t\t";
}

// Implementation of the Arg_intval class
Arg_intval::Arg_intval(const clString& name, int val,
		       const clString& usage_string)
: Arg_base(name, usage_string), val(val), set(0)
{
}

Arg_intval::~Arg_intval()
{
}

int Arg_intval::is_set()
{
    return set;
}

int Arg_intval::value()
{
    return val;
}

int Arg_intval::handle(int argc, char** argv, int& idx)
{
    if(idx >= argc){
	cerr << argv[idx-1] << " must be followed by an integer parameter" << endl;
	return 0; // No argument
    }
    clString next(argv[idx++]);
    if(!next.get_int(val)){
	cerr << argv[idx-1] << " is not an integer" << endl;
	cerr << argv[idx-2] << " must be followed by an integer parameter" << endl;
	return 0; // Bad value
    }
    set=1;
}

void Arg_intval::usage_specific()
{
    cerr << "[integer argument]" << endl << "\t\t\t";
}

// Implementation of the Arg_doubleval class
Arg_doubleval::Arg_doubleval(const clString& name, double val,
			     const clString& usage_string)
: Arg_base(name, usage_string), val(val), set(0)
{
}

Arg_doubleval::~Arg_doubleval()
{
}

int Arg_doubleval::is_set()
{
    return set;
}

double Arg_doubleval::value()
{
    return val;
}

int Arg_doubleval::handle(int argc, char** argv, int& idx)
{
    if(idx >= argc){
	cerr << argv[idx-1] << " must be followed by a real number parameter" << endl;
	return 0; // No argument
    }
    clString next(argv[idx++]);
    if(!next.get_double(val)){
	cerr << argv[idx-1] << " is not an real number" << endl;
	cerr << argv[idx-2] << " must be followed by a real number parameter" << endl;
	return 0; // Bad value
    }
    set=1;
}

void Arg_doubleval::usage_specific()
{
    cerr << "[real number argument]" << endl << "\t\t\t";
}

// Implementation of the Arg_flag class
Arg_flag::Arg_flag(const clString& name, const clString& usage_string)
: Arg_base(name, usage_string), set(0)
{
}

Arg_flag::~Arg_flag()
{
}

int Arg_flag::is_set()
{
    return set;
}

int Arg_flag::handle(int, char**, int&)
{
    set=1;
    return 1;
}

void Arg_flag::usage_specific()
{
    // Nothing needed...
}

