
/*
 *  TCL.h: Interface to TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_project_TCL_h
#define sci_project_TCL_h 1

#include <Classlib/Array1.h>
#include <Classlib/String.h>

class TCLvar;

class TCLArgs {
    Array1<clString> args;
public:
    int have_error;
    int have_result;
    clString string;

    TCLArgs(int argc, char* argv[]);
    ~TCLArgs();
    int count();
    clString operator[](int i);

    void error(const clString&);
    void result(const clString&);
    void append_result(const clString&);
    void append_element(const clString&);

    clString make_list(const clString&, const clString&);
    clString make_list(const clString&, const clString&, const clString&);
    clString make_list(const Array1<clString>&);
};

class TCL {
    Array1<TCLvar*> vars;
protected:
    friend class TCLvar;
    void register_var(TCLvar*);
    void unregister_var(TCLvar*);
public:
    static void initialize();
    static void execute(const clString&);
    static void source_once(const clString&);
    static void add_command(const clString&, TCL*, void*);

    TCL();
    virtual ~TCL();
    virtual void tcl_command(TCLArgs&, void*)=0;

    void reset_vars();
};

#endif
