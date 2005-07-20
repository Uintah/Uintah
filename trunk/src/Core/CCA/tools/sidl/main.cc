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
#include <Core/CCA/tools/sidl/Spec.h>
#include <Core/CCA/tools/sidl/SymbolTable.h>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>

extern int yyparse();
extern FILE* yyin;
extern Specification* parse_spec;


bool doing_cia = false;
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

  for ( unsigned int cnt = 0; cnt < possible_cpps.size(); cnt++ ) {
    if ( stat( possible_cpps[ cnt ], &s ) != -1 ) {
      return possible_cpps[ cnt ];
    }
  }

  std::cerr << "ERROR in: ./SCIRun/src/Core/CCA/tools/sidl/main.cc:\n";
  std::cerr << "Cpp: doesn't seem to exist... bye.\n";

  exit( 1 );
  return 0;
}

char* find_builtin()
{
#ifndef SIDL_BUILTINS
#error SIDL_BUILTINS should point to the directory containing cia.sidl
#endif
  return SIDL_BUILTINS "sidl.sidl";
}

int main(int argc, char* argv[])
{
      extern int yydebug;
      yydebug = 0;
      bool failed = false;
      int nfiles = 0;

  const char* cpp = find_cpp();
  bool done_builtin = false;

  std::string outfile;
  bool emit_header = false;

  SpecificationList specs;

std::cerr <<  "sidl: main fxn\n" << std::endl;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-yydebug") == 0) {
      yydebug = 1;
    } else if (argv[i][0] == '-') {
      std::string arg(argv[i]);
      if (arg == "-o") {
        i++;
        if (i >= argc) {
          std::cerr << "No file specified for -o\n";
          exit(1);
        }
        outfile = argv[i];
      } else if (arg == "-h") {
        emit_header = true;
      } else if (arg == "-cia") {
        doing_cia = true;
      } else if (arg == "-I") {
        i++;
        if (i >= argc) {
          std::cerr << "No file specified for -I\n";
          exit(1);
        }
        foremit = false;
        char* ccabuf = new char[strlen(cpp) + strlen(argv[i]) + 10];
        sprintf(ccabuf, "%s %s", cpp, argv[i]);
        yyin = popen(ccabuf, "r");
        delete[] ccabuf;
        if (!yyin) {
          std::cerr << "Error opening file: " << argv[i] << '\n';
          failed = true;
        }
        if (yyparse()) {
          std::cerr << "Error parsing file: " << argv[i] << '\n';
          failed = true;
        }
        if (pclose(yyin) == -1) {
          perror("pclose");
          failed = true;
        }
        parse_spec->isImport = true;
        specs.add(parse_spec);
        parse_spec = 0;
      } else {
        std::cerr << "Unknown option: " << argv[i] << std::endl;
        exit(1);
      }
    } else {
      if (!done_builtin && !doing_cia) {
        foremit = false;
        char* builtin = find_builtin();
std::cerr << "SIDL builtin=" << builtin << std::endl;
        char* buf = new char[strlen(cpp) + strlen(builtin) + 10];
        sprintf(buf, "%s %s", cpp, builtin);
        yyin = popen(buf, "r");
        delete[] buf;
        if (!yyin) {
          std::cerr << "Error opening file: " << builtin << '\n';
          failed = true;
        }
        if (yyparse()) {
          std::cerr << "Error parsing file: " << builtin << '\n';
          failed = true;
        }
        if (pclose(yyin) == -1) {
          perror("pclose");
          failed = true;
        }
        parse_spec->isBuiltin = true;
        specs.add(parse_spec);
        parse_spec = 0;
        done_builtin = true;
      }

      foremit = true;
      nfiles++;
      char* buf = new char[strlen(cpp) + strlen(argv[i]) + 10];
      sprintf(buf, "%s %s", cpp, argv[i]);
      yyin = popen(buf, "r");
      delete[] buf;
      if (!yyin) {
        std::cerr << "Error opening file: " << argv[i] << '\n';
        failed = true;
      }
      if (yyparse()) {
        std::cerr << "Error parsing file: " << argv[i] << '\n';
        failed = true;
      }
      if (pclose(yyin) == -1) {
        perror("pclose");
        failed = true;
      }
      parse_spec->setTopLevel();
      specs.add(parse_spec);
      parse_spec = 0;
    }
  }
  if (failed) {
    exit(1);
  }
  if (nfiles == 0) {
    std::cerr << "Must specify a file to parse\n";
  }

  /*
   * Process imports...
   */

  specs.processImports();

  /*
   * Static checking
   */
  specs.staticCheck();

  /*
   * Emit code
   */
  std::ofstream devnull("/dev/null");
  if (outfile != "") {
    std::ofstream out(outfile.c_str());
    if (!out) {
      std::cerr << "Error opening output file: " << outfile << '\n';
      exit(1);
    }
    std::string hname = outfile;
    int l = hname.length()-1;
    while(l>0 && hname[l] != '.')
      l--;
    if (l>0)
      hname = hname.substr(0, l);
    hname += ".h";
    if (emit_header)
      specs.emit(devnull, out, hname);
    else
      specs.emit(out, devnull, hname);
  } else {
    std::string hname = "stdout";
    if (emit_header)
      specs.emit(devnull, std::cout, hname);
    else
      specs.emit(std::cout, devnull, hname);
  }
  return 0;
}

