/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


#include <iostream>
#include <stdio.h>
#include "Spec.h"
#include "SymbolTable.h"
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>



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
  struct stat s;
  char * cpp = "/usr/lib/gcc-lib/i586-mandrake-linux/egcs-2.91.66/cpp";

  if( stat( cpp, &s ) == -1 )
    {
      cerr << "ERROR in: ./SCIRun/src/Core/CCA/tools/sidl/main.cc:\n";
      cerr << "Cpp: " << cpp << "doesn't seem to exist... bye.\n";
      exit( 1 );
    }

  return cpp;
  //    return "/usr/lib/gcc-lib/i586-mandrake-linux/egcs-2.91.66/cpp";

  //    return "/usr/lib/gcc-lib/i386-redhat-linux/2.7.2.3/cpp";
  //    return "/usr/lib/cpp";
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

