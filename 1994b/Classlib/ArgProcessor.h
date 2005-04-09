
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

#ifndef SCI_Classlib_ArgProcessor_h
#define SCI_Classlib_ArgProcessor_h 1

#include <Classlib/String.h>
class Arg_base;

class ArgProcessor {
    ArgProcessor();
public:
    static void register_arg(const clString&, Arg_base*);
    static void process_args(int, char**);
    static void get_x_args(int&, char**&);
    static void usage();
    static clString get_program_name();
};

#endif

