
/*
 *  ArgsProcessor.h: Interface to Argument processor class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/ArgProcessor.h>
#include <Classlib/Args.h>
#include <Classlib/AVLTree.h>
#include <Classlib/HashTable.h>
#include <Classlib/Exceptions.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <stdlib.h>

static HashTable<clString, Arg_base*>* known_args=0;
static int x_argc;
static char** x_argv;
static clString progname;

void ArgProcessor::register_arg(const clString& name, Arg_base* handler)
{
    if(!known_args)
	known_args=scinew HashTable<clString, Arg_base*>;
    Arg_base* dummy;
    if(known_args->lookup(name, dummy))
	EXCEPTION(General("Duplicate argument added to argument database"));
    known_args->insert(name, handler);
}

void ArgProcessor::process_args(int argc, char** argv)
{
    int idx=1;
    progname=argv[0];
    x_argc=1;
    x_argv=scinew char*[argc+1];
    x_argv[0]=argv[0];
    while(idx < argc){
	char* arg=argv[idx];
	if(arg[0] == '-'){
	    clString argstring(arg+1);
	    // See if it is an X windows arg...
	    if(argstring == "bg"
	       || argstring == "background"
	       || argstring == "bd"
	       || argstring == "bw"
	       || argstring == "borderwidth"
	       || argstring == "bordercolor"
	       || argstring == "display"
	       || argstring == "fg"
	       || argstring == "fn"
	       || argstring == "font"
	       || argstring == "foreground"
	       || argstring == "geometry"
	       || argstring == "name"
	       || argstring == "title"
	       || argstring == "xrm"){
		x_argv[x_argc++]=argv[idx++];
		if(idx >= argc){
		    cerr << "Argument " << argv[idx-1] << " requires a parameter" << endl;
		    usage();
		} else {
		    x_argv[x_argc++]=argv[idx++];
		}
	    } else if(argstring == "iconic"
		      || argstring == "reverse"
		      || argstring == "rv"
		      || argstring == "synchronous"){
		x_argv[x_argc++]=argv[idx++];
	    } else if(argstring == "help"){
		usage();
	    } else {
		// Look it up in the data base...
		Arg_base* handler;
		if(known_args && known_args->lookup(argstring, handler)){
		    // Handle it...
		    idx++;
		    if(!handler->handle(argc, argv, idx)){
			usage();
		    }
		} else {
		    cerr << "Warning: Argument " << argv[idx] << " not processed" << endl;
		    idx++;
		}
	    }
	} else {
	    cerr << "Warning: Argument " << argv[idx] << " is not a flag" << endl;
	    idx++;
	}
    }
    x_argv[x_argc]=0;
}

void ArgProcessor::get_x_args(int& argc, char**& argv)
{
    argc=x_argc;
    argv=x_argv;
}

void ArgProcessor::usage()
{
    // Alphabetize the known arguments...
    AVLTree<clString, Arg_base*> tree;
    HashTableIter<clString, Arg_base*> hiter(known_args);
    for(hiter.first();hiter.ok();++hiter)
	tree.insert(hiter.get_key(), hiter.get_data());

    cerr << "Usage: " << progname << " [arguments]" << endl;
    cerr << "Possible arguments are:" << endl << endl;
    AVLTreeIter<clString, Arg_base*> iter(&tree);
    for(iter.first();iter.ok();++iter){
	clString name(iter.get_key());
	cerr << "\t-" << name;
	if(name.len() < 16){
	    cerr << "\t";
	    if(name.len() < 8){
		cerr << "\t";
	    }
	} else {
	    cerr << " ";
	}
	iter.get_data()->usage();
	cerr << endl;
    }
    cerr << endl;
    cerr << "All standard X toolkit options are also valid" << endl;
    exit(-1);
}

clString ArgProcessor::get_program_name()
{
    return progname;
}
