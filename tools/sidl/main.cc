
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
    return SIDL_BUILTINS "cia.sidl";
}

int main(int argc, char* argv[])
{
    extern int yydebug;
    yydebug=0;
    bool failed=false;
    int nfiles=0;

    char* cpp=find_cpp();

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
	    } else {
		cerr << "Unknown option: " << argv[i] << endl;
		exit(1);
	    }
	} else {
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
	if(emit_header)
	    specs.emit(devnull, out);
	else
	    specs.emit(out, devnull);
    } else {
	if(emit_header)
	    specs.emit(devnull, std::cout);
	else
	    specs.emit(std::cout, devnull);
    }
    return 0;
}
//
// $Log$
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
