/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



#include <iostream>
#include <stdio.h>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>
#include <Core/CCA/tools/kwai/symTable.h>

extern int yyparse();
extern FILE* yyin;
extern SCIRun::symTable* symT;
using std::cerr;
using std::endl;
using std::string;

bool doing_cia=false;
bool foremit;

using std::vector;

const char* find_cpp()
{
  struct stat s;
  vector<const char*> possible_cpps;

  possible_cpps.push_back( "/usr/lib/gcc-lib/i586-mandrake-linux/egcs-2.91.66/cpp" );
  possible_cpps.push_back( "/usr/lib/gcc-lib/i386-redhat-linux/2.7.2.3/cpp" );
  possible_cpps.push_back( "/usr/lib/cpp" );
  possible_cpps.push_back( "/usr/bin/cpp" );

  for( unsigned int cnt = 0; cnt < possible_cpps.size(); cnt++ ) {
    if( stat( possible_cpps[ cnt ], &s ) != -1 ) {
      return possible_cpps[ cnt ];
    }
  }

  cerr << "ERROR in: ./SCIRun/src/Core/CCA/tools/sidl/main.cc:\n";
  cerr << "Cpp: doesn't seem to exist... bye.\n";

  exit( 1 );
  return 0;
}

int main(int argc, char* argv[])
{
  extern int yydebug;
  yydebug=0;
  bool failed=false;
  int nfiles=0;
  std::string outfile = "out.kwai";
  const char* cpp=find_cpp();

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
  if(nfiles==0){
    cerr << "Must specify a file to parse\n"; exit(1);
  }
  if(failed){
    cerr << "FAILED\n"; exit(1);
  }


  symT->flush(outfile);
  delete symT;

  return 0;
}

