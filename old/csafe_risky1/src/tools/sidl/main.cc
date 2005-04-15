
#include <iostream>
#include <stdio.h>
#include "Spec.h"
#include "SymbolTable.h"
#include <fstream>

extern int yyparse();
extern FILE* yyin;
extern Specification specs;
using std::cerr;
using std::endl;
using std::string;

bool doing_cia=false;
bool foremit;

char* find_cpp()
{
    //return "/usr/lib/gcc-lib/i386-redhat-linux/2.7.2.3/cpp";
    return "/usr/lib/cpp";
}

char* find_builtin()
{
#ifndef SIDL_BUILTINS
#error SIDL_BUILTINS should point to the directory containing cia.sidl
#endif
    return SIDL_BUILTINS "CIA.sidl";
}

int main(int argc, char* argv[])
{
    extern int yydebug;
    yydebug=0;
    bool failed=false;
    int nfiles=0;

    char* cpp=find_cpp();
    bool done_builtin=false;

    std::string outfile;
    bool emit_header=false;

    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-yydebug") == 0){
	    yydebug=1;
	} else if(argv[i][0]=='-'){
	    std::string arg(argv[i]);
	    if(arg == "-o") {
		i++;
		if(i>=argc){
		    cerr << "No file specified for -o\n";
		    exit(1);
		}
		outfile=argv[i];
	    } else if(arg == "-h") {
		emit_header=true;
	    } else if(arg == "-cia") {
		doing_cia=true;
	    } else {
		cerr << "Unknown option: " << argv[i] << endl;
		exit(1);
	    }
	} else {
	    if(!done_builtin && !doing_cia){
		foremit=false;
		char* builtin=find_builtin();
		char* buf=new char[strlen(cpp)+strlen(builtin)+10];
		sprintf(buf, "%s %s", cpp, builtin);
		yyin=popen(buf, "r");
		delete[] buf;
		if(!yyin){
		    cerr << "Error opening file: " << builtin << '\n';
		    failed=true;
		}
		if(yyparse()){
		    cerr << "Error parsing file: " << builtin << '\n';
		    failed=true;
		}
		if(pclose(yyin) == -1){
		    perror("pclose");
		    failed=true;
		}
		done_builtin=true;
	    }

	    foremit=true;
	    nfiles++;
	    char* buf=new char[strlen(cpp)+strlen(argv[i])+10];
	    sprintf(buf, "%s %s", cpp, argv[i]);
	    yyin=popen(buf, "r");
	    delete[] buf;
	    if(!yyin){
		cerr << "Error opening file: " << argv[i] << '\n';
		failed=true;
	    }
	    if(yyparse()){
		cerr << "Error parsing file: " << argv[i] << '\n';
		failed=true;
	    }
	    if(pclose(yyin) == -1){
		perror("pclose");
		failed=true;
	    }
	}
    }
    if(failed){
	exit(1);
    }
    if(nfiles==0){
	cerr << "Must specify a file to parse\n";
    }

    /*
     * Static checking
     */
    specs.staticCheck();

    /*
     * Emit code
     */
    std::ofstream devnull("/dev/null");
    if(outfile != ""){
	std::ofstream out(outfile.c_str());
	if(!out){
	    cerr << "Error opening output file: " << outfile << '\n';
	    exit(1);
	}
	string hname=outfile;
	int l=hname.length()-1;
	while(l>0 && hname[l] != '.')
	    l--;
	if(l>0)
	    hname=hname.substr(0, l);
	hname+= ".h";
	if(emit_header)
	    specs.emit(devnull, out, hname);
	else
	    specs.emit(out, devnull, hname);
    } else {
	string hname="stdout";
	if(emit_header)
	    specs.emit(devnull, std::cout, hname);
	else
	    specs.emit(std::cout, devnull, hname);
    }
    return 0;
}

//
// $Log$
// Revision 1.7  2000/03/23 10:56:03  sparker
// Changed cia.sidl to CIA.sidl
//
// Revision 1.6  1999/09/24 06:26:30  sparker
// Further implementation of new Component model and IDL parser, including:
//  - fixed bugs in multiple inheritance
//  - added test for multiple inheritance
//  - fixed bugs in object reference send/receive
//  - added test for sending objects
//  - beginnings of support for separate compilation of sidl files
//  - beginnings of CIA spec implementation
//  - beginnings of cocoon docs in PIDL
//  - cleaned up initalization sequence of server objects
//  - use globus_nexus_startpoint_eventually_destroy (contained in
// 	the globus-1.1-utah.patch)
//
// Revision 1.5  1999/09/17 05:07:27  sparker
// Added nexus code generation capability
//
// Revision 1.4  1999/09/04 06:00:43  sparker
// Changed place to find cpp
// Updates to cca spec
//
// Revision 1.3  1999/08/30 20:19:29  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
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
