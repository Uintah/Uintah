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

