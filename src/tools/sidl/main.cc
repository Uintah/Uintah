
#include <iostream.h>
#include <stdio.h>
#include "Spec.h"
#include "SymbolTable.h"

extern int yyparse();
extern FILE* yyin;
extern Specification specs;

char* find_cpp()
{
    //    return "/usr/lib/gcc-lib/i386-redhat-linux/2.7.2.3/cpp";
    return "/usr/lib/cpp";
}

char* find_builtin()
{
    return "cia.sidl";
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

    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-yydebug") == 0){
	    yydebug=1;
	} else if(argv[i][0]=='-'){
	    cerr << "Unknown option: " << argv[i] << endl;
	    exit(1);
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

    cerr << "Parsing done\n";
    /*
     * Static checking
     */
    SymbolTable globals(0, "Global Symbols");
    specs.staticCheck(&globals);

    /*
     * Emit Ports
     */
    return 0;
}
