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
#include <fstream>
#include <vector>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <Core/CCA/tools/strauss/strauss.h>

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using namespace SCIRun;


bool fileExists(const char* mFile)
{
  struct stat s;

  if( stat( mFile, &s ) != -1 ) {
    return true;
  }
  return false;
}

int main(int argc, char* argv[])
{
  int status = 0; 
  string pluginSpec;
  string portSpec;
  string header("Bridge.h");
  string impl("Bridge.cc");

  if(argc < 3) {
    cerr << "Wrong parameters to strauss.\n";
    cerr << "USAGE: strauss -p plugin portspec.xml\n"; 
    exit(1);
  }

  for(int i=1;i<argc;i++){
    std::string arg(argv[i]);
    if(arg == "-p") {
      pluginSpec=argv[++i];
      if(!fileExists(pluginSpec.c_str())) {
        cerr << "ERROR: " << pluginSpec << "... doesn't seem to exist... bye.\n";
        exit(1);
      }
    } else if(arg == "-o") {
      header = std::string(argv[++i]) + ".h";
      impl = std::string(argv[i]) + ".cc"; 
    } else {
      portSpec = argv[i];  
      if(!fileExists(portSpec.c_str())) {
        cerr << "ERROR: " << portSpec << "... doesn't seem to exist... bye.\n";
        exit(1);
      }
    }
  }

  try {
    Strauss* strauss = new Strauss(pluginSpec,portSpec,header,impl);
    status = strauss->emit();
    strauss->commitToFiles();
    delete strauss;
  }
  catch (...) {
    cerr << "Strauss: FAILED\n";
    return 1;
  }

  return status;
}

