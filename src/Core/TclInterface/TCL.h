
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

#include <Containers/Array1.h>
#include <Containers/String.h>

namespace SCICore {
namespace TclInterface {

using SCICore::Containers::Array1;
using SCICore::Containers::clString;

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

    static clString make_list(const clString&, const clString&);
    static clString make_list(const clString&, const clString&, const clString&);
    static clString make_list(const Array1<clString>&);
};

class TCL {
    Array1<TCLvar*> vars;
    friend class TCLvar;
    void register_var(TCLvar*);
    void unregister_var(TCLvar*);
public:
    virtual void emit_vars(ostream& out);
    static void initialize();
    static void execute(const clString&);
    static void execute(char*);
    static int eval(const clString&, clString& result);
    static int eval(char*, clString& result);
    static void source_once(const clString&);
    static void add_command(const clString&, TCL*, void*);
    static void delete_command( const clString& command );

    TCL();
    virtual ~TCL();
    virtual void tcl_command(TCLArgs&, void*)=0;

    void reset_vars();

    // To get at tcl variables
    int get_tcl_stringvar(const clString& base, const clString& name,
			  clString& value);
    int get_tcl_boolvar(const clString& base, const clString& name,
			int& value);
    int get_tcl_doublevar(const clString& base, const clString& name,
			  double& value);
    int get_tcl_intvar(const clString& base, const clString& name,
		       int& value);

    void set_tclvar(const clString& base, const clString& name,
		    const clString& value);

};

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:16  mcq
// Initial commit
//
// Revision 1.5  1999/07/07 21:11:03  dav
// added beginnings of support for g++ compilation
//
// Revision 1.4  1999/05/13 18:15:51  dav
// Added back in the virtual on emit_vars
//
// Revision 1.3  1999/05/06 19:56:24  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:34  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
