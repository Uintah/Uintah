
#include <iostream>
#include <stdio.h>
#include <fstream>

extern int yyparse();
extern FILE* yyin;
using std::cerr;
using std::endl;
using std::string;
extern char* curfile;
extern int lineno;

int main(int argc, char* argv[])
{
    extern int yydebug;
    yydebug=0;
    bool failed=false;
    int nfiles=0;

    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-yydebug") == 0){
	    yydebug=1;
	} else {
	    nfiles++;
	    yyin=fopen(argv[i], "r");
	    if(!yyin){
		cerr << "Error opening file: " << argv[i] << '\n';
		failed=true;
	    }
	    curfile=argv[i];
	    lineno=1;
	    if(yyparse()){
		cerr << "Error parsing file: " << argv[i] << '\n';
		failed=true;
	    }
	    if(fclose(yyin) == -1){
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

}

